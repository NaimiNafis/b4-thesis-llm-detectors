import ast
import json
import random

class DeadCodeInserter(ast.NodeTransformer):
    """
    AST transformer to insert dead code (an `if False:` block) into functions.
    """
    def __init__(self):
        super().__init__()
        self.dead_code_counter = 0

    def _create_dead_code_block(self):
        """Helper method to create a new `if False:` AST node."""
        var_name = f"dead_code_var_{self.dead_code_counter}"
        self.dead_code_counter += 1
        
        # Create the assignment statement: dead_code_var_0 = "this code does not run"
        assign_node = ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=ast.Constant(value="this code does not run")
        )
        
        # Create the `if False:` block
        if_false_node = ast.If(
            test=ast.Constant(value=False),
            body=[assign_node],
            orelse=[]
        )
        return if_false_node

    def visit_FunctionDef(self, node):
        # First, visit any nested functions within this one
        self.generic_visit(node)
        
        # We only insert code if the function body has at least one statement
        if node.body and not isinstance(node.body[0], ast.Pass):
            # Create the dead code block
            dead_code_block = self._create_dead_code_block()
            
            # Choose a random position to insert the dead code
            # It can be at the beginning, middle, or end of the function body
            insert_pos = random.randint(0, len(node.body))
            node.body.insert(insert_pos, dead_code_block)
            
        return node

def create_dead_code_attack_dataset(input_file, output_file):
    """
    Reads a .jsonl file, applies the dead code insertion to the 'code' field,
    and writes the result to a new .jsonl file.
    """
    transformer = DeadCodeInserter()

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                original_code = data.get("code", "")

                if original_code:
                    tree = ast.parse(original_code)
                    modified_tree = transformer.visit(tree)
                    # This step ensures line numbers and other metadata are correct
                    ast.fix_missing_locations(modified_tree)
                    modified_code = ast.unparse(modified_tree)
                    data["code"] = modified_code
                
                outfile.write(json.dumps(data) + '\n')

            except (SyntaxError, json.JSONDecodeError) as e:
                print(f"Warning: Could not process line {line_num}. Copying original. Error: {e}")
                outfile.write(line)

    print(f"Finished creating attack dataset B at: {output_file}")

if __name__ == '__main__':
    input_dataset_path = 'test.jsonl'
    output_dataset_path = 'attack_B_dead_code.jsonl'
    create_dead_code_attack_dataset(input_dataset_path, output_dataset_path)