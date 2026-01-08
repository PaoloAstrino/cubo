from pathlib import Path
p = Path('data')
for f in sorted(p.iterdir()):
    if f.is_file():
        name = f.name
        print(repr(name))
        print('codepoints:', [hex(ord(c)) for c in name])
        print('---')
