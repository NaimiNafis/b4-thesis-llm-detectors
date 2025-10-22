import ast
import json
import random

class RenameTransformer(ast.NodeTransformer):
    """
    AST transformer to rename function parameters and local variables to generic names.
    e.g., param_0, param_1, var_0, var_1, etc.
    """
    def __init__(self):
        super().__init__()
        self.rename_map = {}
        self.param_counter = 0
        self.var_counter = 0

    def visit_FunctionDef(self, node):
        # Reset counters and map for each new function to handle scope
        self.rename_map = {}
        self.param_counter = 0
        self.var_counter = 0

        # Rename parameters (arguments)
        for arg in node.args.args:
            old_name = arg.arg
            new_name = f"param_{self.param_counter}"
            self.rename_map[old_name] = new_name
            arg.arg = new_name
            self.param_counter += 1

        # Visit the body of the function to find local variables and rename them
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                old_name = target.id
                if old_name not in self.rename_map:
                    new_name = f"var_{self.var_counter}"
                    self.rename_map[old_name] = new_name
                    target.id = new_name
                    self.var_counter += 1
        
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if node.id in self.rename_map:
            node.id = self.rename_map[node.id]
        return node

class DeadCodeInserter(ast.NodeTransformer):
    """
    AST transformer to insert dead code (an `if False:` block) into functions.
    """
    def __init__(self):
        super().__init__()
        self.dead_code_counter = 0

    def _create_dead_code_block(self):
        var_name = f"dead_code_var_{self.dead_code_counter}"
        self.dead_code_counter += 1
        
        assign_node = ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=ast.Constant(value="this code does not run")
        )
        
        if_false_node = ast.If(
            test=ast.Constant(value=False),
            body=[assign_node],
            orelse=[]
        )
        return if_false_node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.body and not isinstance(node.body[0], ast.Pass):
            dead_code_block = self._create_dead_code_block()
            insert_pos = random.randint(0, len(node.body))
            node.body.insert(insert_pos, dead_code_block)
        return node

def create_combined_attack_dataset(input_file, output_file):
    """
    Reads a .jsonl file, applies both rename and dead code transformations,
    and writes the result to a new .jsonl file.
    """
    rename_transformer = RenameTransformer()
    dead_code_transformer = DeadCodeInserter()

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                original_code = data.get("code", "")

                if original_code:
                    tree = ast.parse(original_code)
                    
                    # Step 1: Apply the rename transformation
                    renamed_tree = rename_transformer.visit(tree)
                    
                    # Step 2: Apply dead code insertion to the already-renamed tree
                    combined_tree = dead_code_transformer.visit(renamed_tree)
                    
                    ast.fix_missing_locations(combined_tree)
                    modified_code = ast.unparse(combined_tree)
                    data["code"] = modified_code
                
                outfile.write(json.dumps(data) + '\n')

            except (SyntaxError, json.JSONDecodeError) as e:
                print(f"Warning: Could not process line {line_num}. Copying original. Error: {e}")
                outfile.write(line)

    print(f"Finished creating attack dataset C at: {output_file}")

if __name__ == '__main__':
    input_dataset_path = 'test.jsonl'
    output_dataset_path = 'attack_C_combined.jsonl'
    create_combined_attack_dataset(input_dataset_path, output_dataset_path)