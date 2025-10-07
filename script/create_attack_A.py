import ast
import json

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
        # Find local variables during assignment (e.g., x = 10)
        for target in node.targets:
            if isinstance(target, ast.Name):
                old_name = target.id
                # If this variable hasn't been renamed yet (i.e., it's not a parameter)
                if old_name not in self.rename_map:
                    new_name = f"var_{self.var_counter}"
                    self.rename_map[old_name] = new_name
                    target.id = new_name
                    self.var_counter += 1
        
        # Also visit the right side of the assignment
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        # Rename variable usage (e.g., print(x))
        if node.id in self.rename_map:
            node.id = self.rename_map[node.id]
        return node

def create_rename_attack_dataset(input_file, output_file):
    """
    Reads a .jsonl file, applies the rename transformation to the 'code' field,
    and writes the result to a new .jsonl file.
    """
    transformer = RenameTransformer()

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Load the JSON object from the line
                data = json.loads(line.strip())
                original_code = data.get("code", "")

                if original_code:
                    # Parse the original code into an AST
                    tree = ast.parse(original_code)
                    
                    # Transform the tree using our custom renamer
                    modified_tree = transformer.visit(tree)
                    
                    # Convert the modified tree back to source code
                    modified_code = ast.unparse(modified_tree)
                    
                    # Update the 'code' field in our data object
                    data["code"] = modified_code
                
                # Write the modified data object as a JSON string to the output file
                outfile.write(json.dumps(data) + '\n')

            except (SyntaxError, json.JSONDecodeError) as e:
                print(f"Warning: Could not process line {line_num}. Copying original. Error: {e}")
                # If something goes wrong, just write the original line to the output
                outfile.write(line)

    print(f"Finished creating attack dataset A at: {output_file}")

if __name__ == '__main__':
    input_dataset_path = 'test.jsonl'
    output_dataset_path = 'attack_A_rename.jsonl'
    create_rename_attack_dataset(input_dataset_path, output_dataset_path)