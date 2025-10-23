import json
import textwrap
import libcst as cst
from libcst.metadata import ParentNodeProvider
from typing import Optional
from .preprocess import preprocess_code
from .transformer import LibCSTRenameTransformer
from .wrappers import attempt_wrappers


def process_file(input_file: str, output_file: str, enable_lib2to3: bool = False, limit: Optional[int] = None):
    debug_file = output_file.replace('.jsonl', '_debug.jsonl')
    processed_count = 0
    error_count = 0
    wrapped_count = 0
    rescued_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile, \
         open(debug_file, 'w', encoding='utf-8') as dbgout:

        for line_num, line in enumerate(infile, 1):
            if limit and processed_count >= limit:
                break
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError as e:
                outfile.write(line)
                error_count += 1
                dbgout.write(json.dumps({"line": line_num, "status": "bad_json", "error": str(e)}) + "\n")
                continue

            original_code = data.get('code', '')
            index = data.get('index')
            if not original_code:
                outfile.write(json.dumps(data) + '\n')
                processed_count += 1
                dbgout.write(json.dumps({"line": line_num, "index": index, "status": "empty_code"}) + "\n")
                continue

            pre = preprocess_code(original_code, enable_lib2to3=enable_lib2to3)

            transformer = LibCSTRenameTransformer()
            try:
                tree = cst.parse_module(pre)
                wrapper = cst.metadata.MetadataWrapper(tree, cache={ParentNodeProvider})
                modified_tree = wrapper.visit(transformer)
                new_code = modified_tree.code
                status = 'ok'
                mapping = transformer.collected
                rescued = True
            except Exception as e:
                # Catch parser errors and CST validation errors and try wrapper fallbacks
                wrapped_count += 1
                status, new_code, mapping, error = attempt_wrappers(pre)
                rescued = status != 'failed'
                if rescued:
                    rescued_count += 1
                else:
                    error = error or str(e)

            if 'rescued' in locals() and rescued:
                data['code'] = new_code
                outfile.write(json.dumps(data) + '\n')
                processed_count += 1
                dbgout.write(json.dumps({"line": line_num, "index": index, "status": status, "mapping": mapping}) + "\n")
            else:
                outfile.write(line)
                processed_count += 1
                error_count += 1
                dbgout.write(json.dumps({"line": line_num, "index": index, "status": "failed", "error": error}) + "\n")

    print(f"\nFinished creating attack dataset A at: {output_file}")
    print(f"Processed (total lines): {processed_count}")
    print(f"Rescued (wrapped & transformed): {rescued_count}")
    print(f"Wrapped attempts: {wrapped_count}")
    print(f"Failed/left unchanged: {error_count} (see {debug_file})")
