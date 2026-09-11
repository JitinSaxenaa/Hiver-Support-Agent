with open('app.py', 'rb') as f:
    content = f.read().decode('utf-8')

import re
# Replace \" with "
fixed = content.replace('\\\"', '\"').replace(\"\\'\", \"'\")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(fixed)

import py_compile
py_compile.compile('app.py', doraise=True)
print('app.py syntax is completely clean and verified!')
