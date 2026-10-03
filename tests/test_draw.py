import pytest

pytest.importorskip("schemdraw")
pytest.importorskip("matplotlib")
import schemdraw.elements as elm

from circuit_quantization import Branch, Circuit, draw_circuit_diagram
from circuits import zero_pi


@pytest.mark.parametrize("reverse", [False, True])
def test_voltage_polarity_is_source_relative_to_ground(reverse):
    ends = ("src", 0) if reverse else (0, "src")
    circuit = Circuit([
        Branch("J", 1, 0, C=1, EJ=1), Branch("Cg", 1, "src", C=1),
        Branch("Vg", *ends, V=1),
    ], ground=0)
    drawing = draw_circuit_diagram(circuit, layout={0: (0, 0), 1: (-1, 1), "src": (0, 1)})
    source = next(e for e in drawing.elements if isinstance(e, elm.SourceV))
    ground = next(e for e in drawing.elements if isinstance(e, elm.Ground))
    assert tuple(source.absanchors["start"]) == (0, 0)
    assert tuple(source.absanchors["end"]) == pytest.approx((0, 4))
    assert tuple(ground.absanchors["start"]) == (0, 0)


def test_parallel_elements_and_inductor_capacitance_have_separate_lanes():
    circuit = Circuit([
        Branch("J", 1, 0, C=1, EJ=1), Branch("L", 0, 1, C=1, L=1),
        Branch("C", 1, 0, C=2),
    ], ground=0)
    drawing = draw_circuit_diagram(circuit)
    symbols = [e for e in drawing.elements if isinstance(e, (elm.Capacitor, elm.Josephson, elm.Inductor2))]
    assert len(symbols) == 4  # The junction's intrinsic capacitance is included in its box.
    assert len({tuple(e.absanchors["center"]) for e in symbols}) == 4


def test_parallel_source_polarity_uses_group_endpoints():
    circuit = Circuit([
        Branch("J", 1, 0, C=1, EJ=1), Branch("Cg", 1, "src", C=1),
        Branch("Cs", "src", 0, C=1), Branch("Vg", 0, "src", V=1),
    ], ground=0)
    drawing = draw_circuit_diagram(circuit, layout={0: (0, 0), 1: (-1, 1), "src": (0, 1)})
    source = next(e for e in drawing.elements if isinstance(e, elm.SourceV))
    assert source.absanchors["end"].y > source.absanchors["start"].y


@pytest.mark.parametrize("layout", ["planar", "spring", "polygon"])
def test_floating_circuit_draws_no_ground(layout):
    drawing = draw_circuit_diagram(zero_pi(), layout=layout)
    assert not any(isinstance(e, elm.Ground) for e in drawing.elements)


@pytest.mark.parametrize("extension", ["svg", "pdf", "png"])
def test_save_diagram(tmp_path, extension):
    circuit = Circuit([Branch("J", 1, 0, C=1, EJ=1)], ground=0)
    out = tmp_path / f"circuit.{extension}"
    drawing = draw_circuit_diagram(circuit, out, label=False, node_labels=False)
    assert drawing is not None
    assert out.stat().st_size > 100


@pytest.mark.parametrize("layout", [{1: (0, 0)}, {0: (0, 0), 1: (0, 0)}, "unknown"])
def test_invalid_layout(layout):
    circuit = Circuit([Branch("J", 1, 0, C=1, EJ=1)], ground=0)
    with pytest.raises(ValueError):
        draw_circuit_diagram(circuit, layout=layout)



def test_notebook_svg_representation():
    circuit = Circuit([Branch("J", 1, 0, C=1, EJ=1)], ground=0)
    svg = draw_circuit_diagram(circuit)._repr_svg_()
    assert "<svg" in svg


def test_flux_labels_use_layout_units_and_render_mathtext():
    circuit = Circuit([Branch("J", 1, 0, C=1, EJ=1)], ground=0)
    drawing = draw_circuit_diagram(circuit, layout={0: (0, 0), 1: (0, 1)}, scale=3,
                                   flux_labels={r"$\odot\,\Phi_k$": (1, .5)}, label=False)
    annotation = next(e for e in drawing.elements if isinstance(e, elm.Label))
    assert tuple(annotation.absanchors["xy"]) == pytest.approx((3, 1.5))
    assert "<svg" in drawing._repr_svg_()


@pytest.mark.parametrize("point", [(1,), (1, float("nan"))])
def test_invalid_flux_label_position(point):
    circuit = Circuit([Branch("J", 1, 0, C=1, EJ=1)], ground=0)
    with pytest.raises(ValueError, match="flux label positions"):
        draw_circuit_diagram(circuit, flux_labels={"Phi": point})
