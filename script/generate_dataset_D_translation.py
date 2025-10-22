import json
import time
import hashlib
import ast
import sqlite3
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import re
try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except Exception:
    # tenacity is optional for retries; fall back to no-op decorator
    def retry(*a, **kw):
        def _dec(f):
            return f
        return _dec

try:
    import openai
except Exception:
    openai = None

class CostCalculator:
    """Calculates API usage costs and token estimates"""
    
    # Current pricing
    PRICING = {
        "openai": {
            # Updated to user-supplied pricing: per 1M tokens input $0.50, output $1.50
            # Convert to per-1k-token rates: input = 0.50*(1k/1e6)=0.0005, output = 1.5*(1k/1e6)=0.0015
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-4": {"input": 0.03, "output": 0.06}
        }
    }
    
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        # tiktoken is optional in this script; if available, use it for better estimates
        try:
            import tiktoken
            if provider == "openai":
                self.encoding = tiktoken.encoding_for_model(model)
            else:
                self.encoding = None
        except Exception:
            self.encoding = None
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the text"""
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception:
                return max(1, len(text) // 4)
        else:
            return max(1, len(text) // 4)
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD"""
        pricing = self.PRICING[self.provider][self.model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

class TranslationCache:
    """Caches translations to avoid redundant API calls"""
    
    def __init__(self, cache_file: str = "translation_cache.db"):
        self.cache_file = cache_file
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.cache_file) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS translations (
                    input_hash TEXT PRIMARY KEY,
                    input_code TEXT,
                    translated_code TEXT,
                    provider TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _get_hash(self, code: str, provider: str) -> str:
        """Create a unique hash for the input code and provider"""
        return hashlib.sha256(f"{code}:{provider}".encode()).hexdigest()
    
    def get(self, code: str, provider: str) -> Optional[str]:
        """Retrieve cached translation"""
        code_hash = self._get_hash(code, provider)
        with sqlite3.connect(self.cache_file) as conn:
            result = conn.execute(
                "SELECT translated_code FROM translations WHERE input_hash = ?",
                (code_hash,)
            ).fetchone()
            return result[0] if result else None
    
    def store(self, input_code: str, translated_code: str, provider: str):
        """Store translation in cache"""
        code_hash = self._get_hash(input_code, provider)
        with sqlite3.connect(self.cache_file) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO translations (input_hash, input_code, translated_code, provider) VALUES (?, ?, ?, ?)",
                (code_hash, input_code, translated_code, provider)
            )

class CodeValidator:
    """Validates translated Python code"""
    
    @staticmethod
    def validate_syntax(code: str) -> Tuple[bool, str]:
        """Check if the code is syntactically valid Python"""
        try:
            ast.parse(code)
            return True, "Valid syntax"
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
    
    @staticmethod
    def validate_semantics(code: str) -> Tuple[bool, str]:
        """Basic semantic validation of the translated code"""
        try:
            tree = ast.parse(code)
            # Check for common translation issues
            issues = []
            
            class TranslationValidator(ast.NodeVisitor):
                def visit_Name(self, node):
                    # Check for common Java naming patterns that shouldn't be in Python
                    if node.id.startswith("get") or node.id.startswith("set"):
                        issues.append(f"Possible Java-style accessor: {node.id}")
                    self.generic_visit(node)
                
                def visit_Call(self, node):
                    # Check for common Java methods that should be Python built-ins
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in ["length", "size"]:
                            issues.append(f"Use len() instead of {node.func.attr}")
                    self.generic_visit(node)
            
            TranslationValidator().visit(tree)
            return len(issues) == 0, "\n".join(issues) if issues else "Valid semantics"
            
        except Exception as e:
            return False, f"Semantic validation error: {str(e)}"

class LLMTranslator:
    """Translator that uses OpenAI's gpt-3.5-turbo API to translate Java -> Python."""

    SYSTEM_PROMPT = (
        "You are a code translator. Translate the given Java code into idiomatic Python. "
        "Only return the translated Python code without explanations. Preserve functionality."
    )

    def __init__(self, api_key: str, model: str = 'gpt-3.5-turbo'):
        openai.api_key = api_key
        self.model = model
        self.cache = TranslationCache()
        self.validator = CodeValidator()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def translate(self, java_code: str) -> Optional[str]:
        # Check cache
        cached = self.cache.get(java_code, 'openai')
        if cached:
            return cached

        prompt = java_code
        try:
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            translated = resp.choices[0].message.content.strip()

            # validate syntax
            ok, msg = self.validator.validate_syntax(translated)
            if not ok:
                print(f"Warning: translated code has syntax issues: {msg}")
                # still cache what we got for inspection
            self.cache.store(java_code, translated, 'openai')
            return translated
        except Exception as e:
            print('OpenAI translation error:', e)
            return None

def create_translation_attack_dataset(
    input_file: str, 
    output_file: str,
    api_key: str,
    provider: str = "openai",
    model: str = "gpt-3.5-turbo",
    batch_size: int = 10
):
    """Creates translation attack dataset with cost estimation and validation"""
    
    translator = LLMTranslator(api_key, model='gpt-3.5-turbo')
    cost_calculator = CostCalculator('openai', 'gpt-3.5-turbo')
    
    total_cost = 0
    processed_count = 0
    success_count = 0
    error_count = 0
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        batch = []
        
        for line in f_in:
            entry = json.loads(line)
            processed_count += 1
            
            try:
                # Translate code
                if translator is None:
                    print('No translator available (openai not installed)')
                    return
                translated_code = translator.translate(entry['code'])
                translated_contrast = translator.translate(entry['contrast'])
                
                if translated_code and translated_contrast:
                    # Calculate actual tokens used
                    input_tokens = cost_calculator.estimate_tokens(entry['code']) + \
                                 cost_calculator.estimate_tokens(entry['contrast'])
                    output_tokens = cost_calculator.estimate_tokens(translated_code) + \
                                  cost_calculator.estimate_tokens(translated_contrast)
                    
                    cost = cost_calculator.calculate_cost(input_tokens, output_tokens)
                    total_cost += cost
                    
                    new_entry = {
                        'index': entry['index'],
                        'code': translated_code,
                        'contrast': translated_contrast,
                        'label': entry['label']
                    }
                    batch.append(new_entry)
                    success_count += 1
                    
                    print(f"Translation {entry['index']} successful. Cost: ${cost:.4f}")
                else:
                    error_count += 1
                    continue
                
                # Save progress after each batch
                if len(batch) >= batch_size:
                    for item in batch:
                        f_out.write(json.dumps(item) + '\n')
                    batch = []
                    print(f"\nProgress Report:")
                    print(f"Processed: {processed_count}")
                    print(f"Successful: {success_count}")
                    print(f"Failed: {error_count}")
                    print(f"Total cost so far: ${total_cost:.4f}\n")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error processing entry {entry['index']}: {str(e)}")
                error_count += 1
                continue
        
        # Write remaining batch
        for item in batch:
            f_out.write(json.dumps(item) + '\n')
        
        # Final report
        print(f"\nFinal Report:")
        print(f"Total processed: {processed_count}")
        print(f"Successful translations: {success_count}")
        print(f"Failed translations: {error_count}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Average cost per translation: ${(total_cost/processed_count):.4f}")

# Example usage:
if __name__ == "__main__":
    api_key = "your-api-key-here"
    
    # First calculate estimated total cost
    calculator = CostCalculator("openai", "gpt-3.5-turbo")
    with open("script/java_test.jsonl", 'r') as f:
        total_lines = sum(1 for _ in f)
        estimated_tokens_per_sample = 1000  # rough estimate
        estimated_total_cost = calculator.calculate_cost(
            total_lines * estimated_tokens_per_sample,
            total_lines * estimated_tokens_per_sample
        )
    
    print(f"Estimated total cost for {total_lines} samples: ${estimated_total_cost:.2f}")
    proceed = input("Do you want to proceed? (y/n): ")
    
    if proceed.lower() == 'y':
        create_translation_attack_dataset(
            input_file="script/java_test.jsonl",
            output_file="script/attack_D_translation.jsonl",
            api_key=api_key,
            provider="openai",
            model="gpt-3.5-turbo"
        )