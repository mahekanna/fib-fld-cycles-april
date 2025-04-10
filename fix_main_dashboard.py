#!/usr/bin/env python3
"""
Script to fix indentation issues in main_dashboard.py
"""

import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_indentation_issues():
    """
    Finds and fixes indentation issues in main_dashboard.py
    """
    filename = "main_dashboard.py"
    
    try:
        # Read the file
        with open(filename, 'r') as f:
            content = f.read()
        
        # Extract the register_callbacks function - this is a robust approach
        # that will rebuild the function structure
        pattern = r"def register_callbacks\(app, scanner, repository\):(.*?)def create_analysis_content"
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            logger.error("Could not find register_callbacks function")
            return False
        
        # Get the function body
        function_body = match.group(1)
        
        # Extract only the top-level variables that should be outside the function
        global_vars = []
        
        # Find the _batch_result_objects definition
        batch_results_pattern = r"# GLOBAL BATCH RESULT STORAGE.*?_batch_result_objects = \[\]"
        batch_results_match = re.search(batch_results_pattern, content, re.DOTALL)
        if batch_results_match:
            global_vars.append(batch_results_match.group(0))
        
        # Find the _strict_consistency_mode definition
        consistency_pattern = r"# Flag to track.*?_strict_consistency_mode = True"
        consistency_match = re.search(consistency_pattern, content, re.DOTALL)
        if consistency_match:
            global_vars.append(consistency_match.group(0))
        
        # Clean up the function body by removing any potential global variables 
        # that were incorrectly included
        for var in global_vars:
            function_body = function_body.replace(var, "")
        
        # Create the new function with proper indentation
        new_function = "def register_callbacks(app, scanner, repository):\n"
        new_function += '    """\n'
        new_function += '    Register all app callbacks.\n'
        new_function += '    \n'
        new_function += '    Args:\n'
        new_function += '        app: Dash application instance\n'
        new_function += '        scanner: FibCycleScanner instance\n'
        new_function += '        repository: ResultsRepository instance\n'
        new_function += '    """\n'
        
        # Add the function body with proper indentation
        for line in function_body.splitlines():
            if line.strip():  # Skip empty lines
                if re.match(r'\s+', line):  # If line is already indented
                    new_function += line + '\n'
                else:  # Add indentation to non-indented lines
                    new_function += '    ' + line + '\n'
            else:
                new_function += '\n'  # Keep empty lines
        
        # Place global variables before the function
        globals_section = "\n".join(global_vars) + "\n\n"
        
        # Replace the original function with the new one, making sure the globals come first
        new_content = re.sub(
            pattern, 
            globals_section + new_function + "def create_analysis_content", 
            content, 
            flags=re.DOTALL
        )
        
        # Write the fixed content back to the file
        with open(filename, 'w') as f:
            f.write(new_content)
            
        logger.info(f"✅ Successfully fixed indentation issues in {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing {filename}: {e}")
        return False

# Apply the fixes
if __name__ == "__main__":
    print("Fixing indentation issues in main_dashboard.py...")
    success = fix_indentation_issues()
    
    if success:
        print("✅ Fixed indentation issues in main_dashboard.py")
        print("You can now run ./extreme_restart.sh again")
    else:
        print("❌ Failed to fix indentation issues")