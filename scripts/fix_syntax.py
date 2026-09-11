import py_compile

with open("app.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("'Directly answering user's specific issue'", '"Directly answering customer specific issue"')

with open("app.py", "w", encoding="utf-8") as f:
    f.write(text)

try:
    py_compile.compile("app.py", doraise=True)
    print("Compilation SUCCESS: app.py compiled with zero errors!")
except Exception as e:
    print("Compilation error:", e)
