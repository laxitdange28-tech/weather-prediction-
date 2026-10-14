import json
import sys
import io
import traceback

def execute_and_populate_notebook(input_path="model.ipynb", output_path="model.ipynb"):
    with open(input_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    global_scope = {}
    execution_counter = 1

    print(f"Executing cells in {input_path}...")

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code_text = "".join(cell["source"])
            old_stdout = sys.stdout
            captured_stdout = io.StringIO()
            sys.stdout = captured_stdout

            try:
                # Execute in shared global scope
                exec(code_text, global_scope)
                output_str = captured_stdout.getvalue()
                cell["execution_count"] = execution_counter
                cell["outputs"] = []

                if output_str.strip():
                    cell["outputs"].append({
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [line + "\n" for line in output_str.splitlines()]
                    })
                print(f"Cell {i} executed successfully.")
            except Exception as e:
                output_str = captured_stdout.getvalue()
                err_msg = traceback.format_exc()
                cell["execution_count"] = execution_counter
                cell["outputs"] = [
                    {
                        "ename": type(e).__name__,
                        "evalue": str(e),
                        "output_type": "error",
                        "traceback": err_msg.splitlines()
                    }
                ]
                print(f"Cell {i} had error: {e}")
            finally:
                sys.stdout = old_stdout
                execution_counter += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Finished executing and saved populated notebook to {output_path}!")

if __name__ == "__main__":
    execute_and_populate_notebook()

