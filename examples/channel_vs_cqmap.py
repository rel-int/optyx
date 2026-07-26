import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Two implementations of classical-quantum maps

        `optyx.channel` and `discopy.quantum.channel` both implement the
        classical-quantum processes of Coecke and Kissinger: completely
        positive maps that carry classical wires alongside quantum ones.
        They agree on the mathematics and disagree on what a morphism *is*.

        * In **discopy**, a `Channel` is a `Tensor`: an array together with a
          pair of dimensions. It is a semantic category — you land in it by
          applying a functor to a circuit, and the result is already a
          number.
        * In **optyx**, a `Channel` is a `Box` in a free frobenius category:
          a name, a domain, a codomain, and a Kraus map stored as another
          diagram. It is a syntactic category — `double()` takes you to
          another syntax, and evaluation is a separate step.

        This notebook runs the same experiments through both and shows where
        the two designs pull apart.
        """
    )
    return


@app.cell
def _():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    import marimo as mo

    from discopy.quantum import Ket, H, Measure as DiscopyMeasure
    from discopy.quantum.channel import Channel as DiscopyChannel, CQ, C
    from discopy.tensor import Dim

    from optyx import classical, photonic, qubits
    from optyx.channel import (
        Channel,
        CQMap,
        Discard,
        Measure,
        bit,
        mode,
        qmode,
        qubit,
    )
    from optyx.core import diagram, zw

    matplotlib.use("Agg")

    def draw(d, **params):
        """Draw a diagram and hand the figure back to marimo."""
        d.draw(show=False, **params)
        return plt.gcf()

    return (
        C,
        CQ,
        CQMap,
        Channel,
        DiscopyChannel,
        DiscopyMeasure,
        Dim,
        Discard,
        H,
        Ket,
        Measure,
        bit,
        classical,
        diagram,
        draw,
        mo,
        mode,
        np,
        photonic,
        qmode,
        qubit,
        qubits,
        zw,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The same experiment, twice

        Prepare $|0\rangle$, apply a Hadamard, measure in the computational
        basis. Both libraries should return the uniform distribution.
        """
    )
    return


@app.cell
def _(DiscopyMeasure, H, Ket):
    discopy_circuit = Ket(0) >> H >> DiscopyMeasure()
    discopy_circuit.eval(mixed=True)
    return (discopy_circuit,)


@app.cell
def _(qubits):
    optyx_circuit = qubits.Ket(0) >> qubits.H() >> qubits.Measure(1)
    optyx_circuit.eval().prob_dist()
    return (optyx_circuit,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Same numbers, different route. `eval(mixed=True)` applies
        `discopy.quantum.channel.Functor`, which maps every box to an array
        and contracts as it goes: the value you get back *is* the object of
        the semantic category. `optyx`'s `eval` first rewrites the diagram
        into the doubled Kraus syntax and only then hands it to a numerical
        backend (quimb by default, with perceval and discopy's tensor
        available too).

        The intermediate step is visible:
        """
    )
    return


@app.cell
def _(optyx_circuit):
    optyx_circuit.double()
    return


@app.cell
def _(draw, optyx_circuit):
    draw(optyx_circuit.double(), figsize=(5, 4))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The doubled diagram is still a diagram — an `optyx.core.diagram.Diagram`
        over the pure generators. Each quantum wire has become two wires (the
        state and its conjugate), and the measurement has become a spider that
        fuses them into one classical wire. In discopy the corresponding step
        is `Channel.double(tensor)`, which computes
        `array ⊗ conjugate(array)` and returns numbers; there is no doubled
        diagram to look at.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Objects: dimensions versus generators

        A discopy `CQ` is a pair of dimensions, and the underlying tensor
        shape is `classical @ quantum @ quantum` — the quantum part is
        squared because density matrices are doubled.
        """
    )
    return


@app.cell
def _(CQ, Dim):
    CQ(Dim(2), Dim(3)), CQ(Dim(2), Dim(3)).to_dim()
    return


@app.cell
def _(bit, qmode, qubit):
    optyx_type = qubit @ bit @ qmode
    optyx_type, optyx_type.single(), optyx_type.double()
    return (optyx_type,)


@app.cell
def _(mo):
    mo.md(
        r"""
        An optyx `Ty` is a list of generators drawn from `bit`, `mode`,
        `qubit` and `qmode`. `single()` forgets the quantum/classical
        distinction and gives the Kraus-level type; `double()` gives the
        type of the CP map, doubling the quantum generators only. So the
        classical/quantum split is recorded per wire rather than as one
        classical block and one quantum block, and the *dimension* is not
        recorded at all.

        That omission is the point. `mode` is a bosonic Fock space: its
        dimension is unbounded, and the truncation is a property of the
        evaluation, not of the diagram. A `CQ` pair of `Dim`s has to commit
        to a number before you can write the map down.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        A wrinkle we hit while writing this notebook: discopy prints a purely
        classical type as if it were quantum, because the third branch of
        `CQ.__str__` returns `Q(self.classical)` instead of
        `C(self.classical)`. The `repr` is correct, so this is cosmetic, but
        it makes the output of `eval(mixed=True)` above misleading — the
        codomain of a measured circuit is classical.
        """
    )
    return


@app.cell
def _(C, Dim, discopy_circuit):
    str(C(Dim(2))), repr(C(Dim(2))), discopy_circuit.eval(mixed=True).cod
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Arrows: arrays versus syntax

        discopy's `Channel` inherits from `Tensor`, so composition is
        contraction and two channels are equal when their arrays are equal.
        optyx's `Channel` inherits from `frobenius.Box`, so composition
        builds a term and two channels are equal when they are the same term.
        """
    )
    return


@app.cell
def _(DiscopyChannel, qubits):
    (
        [c.__name__ for c in DiscopyChannel.__mro__[:4]],
        [c.__name__ for c in type(qubits.H()).__mro__[:4]],
    )
    return


@app.cell
def _(qubits):
    hadamard = qubits.H()
    hadamard.kraus, hadamard.env, hadamard.double()
    return (hadamard,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Every optyx `Channel` carries its Kraus map and an environment type
        `env`. A channel is pure when `env` is empty; discarding the
        environment is what makes it mixed, and `double()` builds that
        discarding explicitly by capping the environment wires with a spider.
        Purity is therefore structural — `is_pure` walks the layers of a
        diagram and looks at the generators — and a pure diagram can be
        turned back into a single Kraus map:
        """
    )
    return


@app.cell
def _(optyx_circuit, qubits):
    pure = qubits.Ket(0) >> qubits.H()
    (pure.is_pure, pure.get_kraus()), (optyx_circuit.is_pure,)
    return


@app.cell
def _(DiscopyMeasure, H, Ket):
    (Ket(0) >> H).is_mixed, (Ket(0) >> H >> DiscopyMeasure()).is_mixed
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        discopy records the same distinction as a boolean, `is_mixed`, and
        the functor branches on the *class* of each box: `Discard`,
        `Measure`, `MixedState`, `Encode` and `Scalar` each get their own
        `isinstance` case inside `channel.Functor.__call__`, and anything
        else is doubled or passed through depending on `is_mixed`. The two
        designs make opposite trade-offs here: optyx puts the data (a Kraus
        map, an environment) on the box, so one rule doubles every box that
        has a Kraus map; discopy keeps boxes free of that data and pays for
        it with a case analysis in the functor.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Where they stop agreeing

        Here is a lossy optical channel: one mode in, one mode out, with a
        Kraus map that splits the amplitude between the mode we keep and an
        environment mode we discard.
        """
    )
    return


@app.cell
def _(Channel, diagram, np, qmode, zw):
    efficiency = 0.8
    loss_kraus = zw.W(2) >> zw.Endo(np.sqrt(efficiency)) @ zw.Endo(
        np.sqrt(1 - efficiency)
    )
    loss = Channel(
        f"loss({efficiency})",
        loss_kraus,
        dom=qmode,
        cod=qmode,
        env=diagram.mode,
    )
    loss.env, loss.kraus.dom, loss.kraus.cod
    return (efficiency, loss)


@app.cell
def _(Measure, loss, photonic, qmode):
    lossy_photon = photonic.Create(1) >> loss >> Measure(qmode)
    lossy_photon.eval().prob_dist()
    return (lossy_photon,)


@app.cell
def _(draw, lossy_photon):
    draw(lossy_photon.double(), figsize=(5, 4))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Nothing in this diagram says how many photons a mode may hold. The
        backend picks a truncation from the diagram when it contracts. To
        write the same channel as a discopy `Channel` you would first have to
        choose a Fock cutoff, build the array by hand, and rebuild it every
        time the cutoff changes — the CQ types are dimensions, and a photonic
        experiment does not come with one.

        The trade goes the other way too. A discopy `Channel` is a concrete
        CP map, so you can add channels, compare them numerically, and read
        off the density matrix without leaving the category. optyx has a box
        for that direction, `CQMap`, which takes the doubled diagram instead
        of a Kraus map — it is the closest thing in optyx to a discopy
        `Channel`, and it is what the classical boxes and the
        photon-gathering step of `Measure.inflate` are built from, since
        neither has a natural Kraus decomposition.
        """
    )
    return


@app.cell
def _(CQMap, classical):
    parity = classical.X(2, 1)
    isinstance(parity, CQMap), parity.density_matrix, parity.dom, parity.cod
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        | | `discopy.quantum.channel` | `optyx.channel` |
        |---|---|---|
        | objects | `CQ`: a pair of dimensions | `Ty` over `bit`, `mode`, `qubit`, `qmode` |
        | arrows | arrays (`Tensor` subclass) | boxes and diagrams (`frobenius` subclass) |
        | equality | numerical | syntactic |
        | doubling | on arrays, inside the functor | on syntax, via `Diagram.double()` |
        | purity | `is_mixed` flag plus `isinstance` cases in the functor | `env` type on every box, `is_pure` reads the layers |
        | dimensions | fixed in the type | chosen by the backend |
        | evaluation | is the semantics | a separate step, with a choice of backends |
        | infinite-dimensional systems | need a cutoff up front | native (`mode`, `qmode`) |
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Which one I prefer, and how I would unify them

        I prefer optyx's, but only because of what it is for. Keeping
        channels syntactic and pushing doubling into a functor between two
        free categories means every intermediate object is still a diagram
        you can draw, rewrite, differentiate and hand to a different backend,
        and it lets `mode` be a genuine Fock space instead of a dimension
        chosen in advance — which is the whole reason optyx exists. What I do
        not prefer is the part optyx carries that is not about photonics: the
        Kraus-map-plus-environment data on every box is a design decision,
        and it currently coexists with `CQMap` and with per-class overrides
        of `double`, `inflate` and `dagger`, so there are three ways to say
        what a box means. discopy's version is smaller and its story is
        cleaner — a `CQ` object, an array, a functor from circuits — and it
        pays for that with a case analysis over box classes and with no
        answer for unbounded dimension.

        The unification I would like is to make discopy's channel category
        the *semantics* of optyx's, rather than a parallel implementation.
        Concretely: give discopy's `Channel` a `Dim`-free companion (or let
        `CQ` hold symbolic dimensions), then define optyx's evaluation as a
        functor into it, so that `Diagram.double() >> to_tensor()` factors as
        "free channel syntax → CQ channels" with the cutoff supplied as a
        functor parameter rather than baked into types. Upstream in discopy,
        the `isinstance` chain in `channel.Functor.__call__` would become a
        method on the boxes — each box says how it doubles, which is what
        optyx's `Channel.double` already does — and optyx would drop its own
        copy of the CQ bookkeeping and keep only what is photonic: `mode`,
        `qmode`, inflation, and the backends. That would leave one definition
        of a classical-quantum map, one place where doubling is implemented,
        and optyx's channel layer reduced to the free category over it.
        """
    )
    return


if __name__ == "__main__":
    app.run()
