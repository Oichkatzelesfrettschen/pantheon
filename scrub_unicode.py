import os
import re

# Mapping of Unicode to ASCII equivalents
SCRUB_MAP = {
    # Emojis
    "[LAUNCH]": "[LAUNCH]", "[BOOM]": "[BOOM]", "[OBS]": "[OBS]", "[NEW]": "[NEW]", 
    "[EXP]": "[EXP]", "[OK]": "[OK]", "[DATA]": "[DATA]", "[COSMO]": "[COSMO]", 
    "[X]": "[X]", "[WARN]": "[WARN]", "( )": "( )", "[X]": "[X]", "[CONFIG]": "[CONFIG]",
    "[PLOT]": "[PLOT]", "[STAT]": "[STAT]", "[LINK]": "[LINK]", "[CALC]": "[CALC]", 
    "[FIT]": "[FIT]", "[LOG]": "[LOG]", "[DEPS]": "[DEPS]", "[OK]": "[OK]",
    "*": "*", "->": "->", "in": "in", "~": "~", "+/-": "+/-",
    "^2": "^2", "^3": "^3", "^4": "^4", "_0": "_0", "_1": "_1", "_n": "_n",
    "pi": "pi", "epsilon": "epsilon", "gamma": "gamma", "chi": "chi", "Omega": "Omega",
    "Lambda": "Lambda", "rho": "rho", "omega": "omega", "nabla": "nabla", "d": "d",
    "integral": "integral", "inf": "inf", "<=": "<=", ">=": ">=", "alpha": "alpha",
    "beta": "beta", "delta": "delta", "sigma": "sigma", "phi": "phi", "lambda": "lambda",
    "eta": "eta", "kappa": "kappa", "mu": "mu", "tau": "tau", "theta": "theta",
    "nu": "nu", "hbar": "hbar", "Sun": "Sun", " deg": " deg",
    
    # Box Drawing
    "+": "+", "=": "=", "+": "+", "|": "|", "+": "+", "+": "+",
    "+": "+", "+": "+", "-": "-", "+": "+", "+": "+", "+": "+",
    "+": "+", "|": "|", "|": "|", "+": "+", "+": "+", "+": "+",
    "#": "#", "_": "_", "^": "^",
}

def scrub_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = content
        for uni, ascii_rep in SCRUB_MAP.items():
            new_content = new_content.replace(uni, ascii_rep)
            
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Scrubbed: {filepath}")
    except Exception as e:
        print(f"Error scrubbing {filepath}: {e}")

def run_audit():
    print("Starting Unicode Audit and Scrub...")
    for root, dirs, files in os.walk('.'):
        if '.git' in dirs:
            dirs.remove('.git')
        for file in files:
            if file.endswith(('.py', '.md', '.txt', '.tex', 'Makefile', 'toml')):
                scrub_file(os.path.join(root, file))

if __name__ == "__main__":
    run_audit()
