# API reference

The top-level package re-exports the names below, so `import circuit_quantization as cq` gives `cq.Circuit`, `cq.quantize`, and so on.

## Circuits and loops

```{eval-rst}
.. automodule:: circuit_quantization.circuit
   :members:
.. automodule:: circuit_quantization.loops
   :members:
```

## Quantization

```{eval-rst}
.. automodule:: circuit_quantization.quantize
   :members:
.. automodule:: circuit_quantization.lagrangian
   :members:
.. automodule:: circuit_quantization.hamiltonian
   :members:
.. automodule:: circuit_quantization.flux
   :members:
```

## Coordinates and normal modes

```{eval-rst}
.. automodule:: circuit_quantization.modes
   :members:
.. automodule:: circuit_quantization.numerical
   :members:
.. automodule:: circuit_quantization.shear
   :members:
```

## Interchange, export, and drawing

```{eval-rst}
.. automodule:: circuit_quantization.io
   :members:
.. automodule:: circuit_quantization.qpackage_interface
   :members:
.. automodule:: circuit_quantization.draw
   :members:
.. automodule:: circuit_quantization.units
   :members:
```

## Extensions

These modules are imported explicitly and are not part of the top-level namespace.

### Symbolic coordinate optimizer

Install with `pip install ".[optimize]"` and put the [Singular](https://www.singular.uni-kl.de) executable on `PATH`.

```{eval-rst}
.. automodule:: circuit_quantization.optimize
   :members:
```

### Product-state solver (experimental)

`circuit_quantization.product_state` is experimental and may change.
The top-level package re-exports only `NumericHamiltonian`, `numeric_hamiltonian`, and `normal_mode_transformation`.

```{eval-rst}
.. automodule:: circuit_quantization.product_state
   :members:
```
