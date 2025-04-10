#\!/usr/bin/env python3

import re
import sys

def fix_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove the orphaned except block
    fixed_content = re.sub(r'    except Exception as e:\s+logger\.exception\(f"Error calculating performance metrics: {str\(e\)}"\)\s+return html\.Div\(\[\s+html\.H4\("Performance Calculation Error"\),\s+html\.P\(f"An error occurred: {str\(e\)}"\)\s+\]\), {"display": "block"}', '', content)
    
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed {file_path}")

if __name__ == "__main__":
    fix_file("/home/vijji/advanced_fld_cycles/fib_cycles_system/web/trading_strategies_ui.py")
