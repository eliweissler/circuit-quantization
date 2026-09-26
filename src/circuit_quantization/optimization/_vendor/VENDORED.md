# Vendored polynomial optimizer

These files were copied from `sircuitenum` commit
`eb67c34e55c718310fb8f0e64991d4890ff8611e` under GPL-3.0:

- `quantize.py` (`789e3280adc6aee91e748feaf932cc3c3199152b660a2a51a861b222027888f5`)
- `equationset.py` (`72fd3ce7b38cc0fcebfffb84977fc2b710e4a11e5c7a53def6544bc30fe0b7cf`)
- `singular_interface.py` (`c5ab57d229dac8b776dbc43e73c9536519d1a2ebfb7b3ceaa45a885aa024bf78`)
- `z3_interface.py` (`49a9389176977c4f0f86702d4c70ad5659c4a09d9001f39d0b7e3df272187730`)
- `rational_check.py` (`fb7c0ab24785f395082582f55d68ec45364e9f181950728ad0075f03cefaee85`)
- `utils.py` (`0d292d18190cd8bf73f59fe1def073829aaedfd7e72883190975d3d289ea3a50`)
- `poly_solver.sing` (`6501fa924e5bdc8372f65e8722564386288b4d69e6efa434c977e2a6603c0a89`)

The Python solver code is unchanged except for mechanical `sircuitenum.*`
import-prefix rewrites needed to keep the copy inside this private namespace.
The branch-centric adapter is maintained separately in the parent package.
