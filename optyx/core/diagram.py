"""

Overview
--------

Optyx diagrams combine three diagrammatic calculi:

- :class:`zw` calculus: for infinite-dimensional systems (Mode type), \
with generators :class:`zw.Z`, :class:`zw.W`, creations and selections.
- :class:`lo` calculus: for linear optics (Mode type), with generators \
:class:`lo.BS` and :class:`lo.Phase`, or other .
- :class:`zx` calculus: for qubit systems (Bit type), with generators \
:class:`zx.Z` and :class:`zx.X`.

Mode and Bit types can moreover be combined using :class:`DualRail`
or other instances of :class:`diagram.core.Box`.
Note that the permanent method is only defined for a subclass
of :class:`zw` diagrams, including :class:`lo` circuits.
These are also known as QPath diagrams [FC23]_,
or matrices with creations and annihilation.
They are implemented in the :class:`path.Matrix` class,
with an interface :class:`to_perceval`
or the internal evaluation method :class:`eval`.
The DisCoPy class :class:`tensor.Diagram` is used as an
implementation of tensor networks,
with dimensions as types and tensors as boxes,
with an interface :class:`to_quimb`
or the internal evaluation method :class:`eval`.
Linear optical circuits, built from the generators of :class:`lo`,
can be evaluated as tensor networks
by first applying the method :class:`to_zw`.


Types
-------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Mode
    Bit
    Ty

Generators and diagrams
------------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Swap
    Scalar
    DualRail
    Feedback

Other classes
-------------
.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    EmbeddingTensor
    FeedbackBoundary
    Stream

Functions
----------
.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    dual_rail
    embedding_tensor

Examples of usage
------------------

**Creating diagrams**

We can create and draw Optyx diagrams using the syntax of
the Discopy package [FTC21]_. Sequential composition of
boxes is done using the :code:`<<` operator:

>>> from optyx.core.zw import Create, W
>>> split_photon = Create(1) >> W(2)
>>> split_photon.draw(path="docs/_static/seq_comp_example.png")

.. image:: /_static/seq_comp_example.png
    :align: center

We can also compose boxes in parallel (tensor) using the :code:`@` operator :

>>> from optyx.photonic import BS, Phase
>>> beam_splitter_phase = (BS @ Phase(0.5)).get_kraus()
>>> beam_splitter_phase.draw(path="docs/_static/parallel_comp_example.png")

.. image:: /_static/parallel_comp_example.png
    :align: center

A beam-splitter from the :class:`photonic` calculus can be
expressed using the :class:`zw` calculus:

>>> from optyx.photonic import BS
>>> beam_splitter = BS.get_kraus()
>>> beam_splitter.draw(path="docs/_static/bs_zw.png")

.. image:: /_static/bs_zw.png
    :align: center

Optyx diagrams can combine the generators from
:class:`zw` (Mode type),
:class:`photonic` (Mode type) and :class:`zx` calculi (Bit type).
We can check their equivalence as tensors.

**Branching Law**

Let's check the branching law from [FC23]_.

>>> from optyx.core.zw import Create, W
>>> from optyx.utils.misc import compare_arrays_of_different_sizes
>>> branching_l = Create(1) >> W(2)
>>> branching_r = Create(1) @ Create(0) + Create(0) @ Create(1)

>>> assert compare_arrays_of_different_sizes(\\
...     branching_l.to_tensor().eval().array,\\
...     branching_r.to_tensor().eval().array,\\
... )

**Hong-Ou-Mandel Effect**

The :code:`to_tensor` method supports evaluation of
diagrams like the Hong-Ou-Mandel effect:

>>> from optyx.core.zw import ZBox, SWAP, W, Select, Id
>>> Zb_i = ZBox(1,1,np.array([1, 1j/(np.sqrt(2))]))
>>> Zb_1 = ZBox(1,1,np.array([1, 1/(np.sqrt(2))]))
>>> beam_splitter = W(2) @ W(2) >> \\
...               Zb_i @ Zb_1 @ Zb_1 @ Zb_i >> \\
...               Id(1) @ SWAP @ Id(1) >> \\
...               W(2).dagger() @ W(2).dagger()
>>> Hong_Ou_Mandel = Create(1) @ Create(1) >> \\
...                beam_splitter >> \\
...                Select(1) @ Select(1)
>>> assert compare_arrays_of_different_sizes(\\
...             Hong_Ou_Mandel.to_tensor().eval().array,\\
...             np.array([0]))

**Permanent evaluation for QPath diagrams**

The :code:`to_path` method supports evaluation by
calculating a permanent of an underlying matrix:

>>> from optyx.core.zw import Create, W
>>> counit_l = W(2) >> Select(0) @ Id(Mode(1))
>>> counit_r = W(2) >> Id(Mode(1)) @ Select(0)
>>> assert counit_l.to_path().eval(2) == counit_r.to_path().eval(2)

References
-----------
.. [FC23] de Felice, G., & Coecke, B. (2023). Quantum Linear Optics via \
String Diagrams. In Proceedings 19th International Conference on \
Quantum Physics and Logic, Wolfson College, Oxford, UK, \
27 June - 1 July 2022 (pp. 83-100). Open Publishing Association.
.. [KW20] Kissinger, A., & Wetering, J. (2020). PyZX: Large Scale \
Automated Diagrammatic Reasoning. In  Proceedings 16th \
International Conference on Quantum Physics and Logic, \
Chapman University, Orange, CA, USA., 10-14 June 2019 \
(pp. 229-241). Open Publishing Association.
.. [Gray18] Gray, J. (2018). quimb: A python package \
for quantum information and many-body calculations. \
Journal of Open Source Software, 3(29), 819.
.. [FGL+23] Heurtel, N., Fyrillas, A., Gliniasty, G., \
Le Bihan, R., Malherbe, S., Pailhas, M., Bertasi, E., \
Bourdoncle, B., Emeriau, P.E., Mezher, R., Music, L., \
Belabas, N., Valiron, B., Senellart, P., Mansfield, S., \
& Senellart, J. (2023). Perceval: A Software Platform \
for Discrete Variable Photonic Quantum Computing. Quantum, 7, 931.
.. [FTC21] de Felice, G., Toumi, A., & Coecke, B. (2021). \
DisCoPy: Monoidal Categories in Python. In  Proceedings Z \
of the 3rd Annual International Applied Category Theory \
Conference 2020,  Cambridge, USA, 6-10th July 2020 (pp. \
183-197). Open Publishing Association.
.. [FSP+23] de Felice, G., Shaikh, R., Poór, B., Yeh, L., \
Wang, Q., & Coecke, B. (2023). Light-Matter Interaction \
in the ZXW Calculus. In  Proceedings of the Twentieth \
International Conference on Quantum Physics and Logic, \
Paris, France, 17-21st July 2023 (pp. 20-46). \
Open Publishing Association.
.. [FPY+24] de Felice, G., Poór, B., Yeh, L., & \
Cashman, W. (2024). Fusion and flow: formal protocols to \
reliably build photonic graph states. arXiv \
preprint arXiv:2409.13541.
"""

from __future__ import annotations

import numpy as np
from sympy.core import Symbol, Mul
from discopy import (
    monoidal, symmetric, frobenius, tensor, hypergraph
)
from discopy import stream as monoidal_stream
from discopy.cat import factory, rsubs
from discopy.frobenius import Dim
from discopy.utils import AxiomError
from discopy.quantum.gates import format_number
from enum import Enum
from optyx.utils.misc import (
    BasisTransition,
    calculate_right_offset,
    get_max_dim_for_box
)
from typing import List, Tuple, Iterable, NamedTuple

MAX_DIM = 10


class PhotonNumberPreservation(Enum):
    """This is used as a flag to indicate how a box acts
    on the incoming photons. Used by the tensor network building
    algorithm to determine the bond dimensions / truncations
    based on number of photons in the past light
    cone of the Box to minimise the truncation dimensions.

    LO: Linear Optical - preserves photon number between input and output
    NON_LO: Does not preserve photon number between input and output
    CUSTOM: Custom behaviour defined by the user
                (a box need to implement a method)
    QUBIT: Qubit - does not act on photons
    """

    LO = "lo"
    NON_LO = "non_lo"
    CUSTOM = "custom"
    QUBIT = "qubit"


class Ob(frobenius.Ob):
    """Basic object in an optyx Diagram: bit or mode"""


@factory
class Ty(frobenius.Ty):
    """Optical and (qu)bit types.

    Equality is structural rather than nominal: a :class:`Mode` or
    :class:`Bit` equals the plain :class:`Ty` with the same objects,
    mirroring :class:`discopy.monoidal.PRO`.
    """

    generator_factory = Ob

    def __eq__(self, other):
        return isinstance(other, Ty) and self.inside == other.inside\
            and (self.dom, self.cod) == (other.dom, other.cod)

    def __hash__(self):
        return hash((Ty, self.inside, self.dom, self.cod))


class Mode(Ty):
    """Optical mode interpreted as the infinite space with countable basis"""

    # pylint: disable=invalid-name
    def __init__(self, n=0):
        self.n = n
        super().__init__(*["mode" for _ in range(n)])


class Bit(Ty):
    """Qubit type interpreted as the two dimensional complext vector space"""

    # pylint: disable=invalid-name
    def __init__(self, n=0):
        self.n = n
        super().__init__(*["bit" for _ in range(n)])


class FeedbackBoundary(NamedTuple):
    """Boundary data attached to one feedback loop.

    The boundary is the memory wire only: ``initial_state`` is plugged into
    it before the first time step and ``final_effect`` onto it after the
    last, with ``mem`` the type carried between two steps. The inputs and
    outputs occurring at every step are not boundaries — they are the domain
    and codomain of the stream at each tick, which stay open.

    The entries are ordered by the functorial traversal used by
    :meth:`Diagram.stream`: left to right for composition, with an outer
    loop before any loop nested in its argument.

    The construction is the delayed feedback of a monoidal stream, as in
    Di Lavore, de Felice and Román, *Monoidal Streams for Dataflow
    Programming* (LICS 2022), which is what ``discopy.monoidal.Stream``
    implements. The `mem` wire is the delay, so this is feedback rather than
    a trace in the sense of Katis, Sabadini and Walters, *Feedback, trace and
    fixed-point semantics* (2002): the loop is not closed by a fixed point,
    it is unrolled one step at a time.
    """
    initial_state: object | None
    final_effect: object | None
    mem: Ty


class Stream(NamedTuple):
    """A diagram's stream and its ordered feedback boundaries."""
    stream: monoidal_stream.Stream
    boundaries: tuple[FeedbackBoundary, ...]


@factory
class Diagram(frobenius.Diagram):
    """Optyx diagram combining :class:`zw`,
    :class:`zx` and
    :class:`lo` calculi."""

    grad = tensor.Diagram.grad

    def conjugate(self) -> Diagram:
        """Conjugates every box in the diagram"""
        return symmetric.Functor(
            ob_map=lambda x: x,
            ar_map=lambda f: f.conjugate(),
            cod=Diagram,
            dom=Diagram,
        )(self)

    def to_path(self, dtype: type = complex):
        """Returns the :class:`Matrix` normal form
        of a :class:`Diagram`.
        In other words, it is the underlying matrix
        representation of a :class:`path` and :class:`lo` diagrams.

        Feedback loops have no path semantics: they raise, asking for the
        diagram to be unrolled first.
        """
        # pylint: disable=import-outside-toplevel
        from optyx.core import path

        return symmetric.Functor(
            ob_map=len,
            ar_map=lambda f: f.to_path(dtype),
            cod=path.Matrix[dtype],
        )(self)

    def feedback(self, dom=None, cod=None, mem=None,
                 initial_state=None, final_effect=None) -> Diagram:
        """
        Feed the last `mem` outputs of a diagram from `dom @ mem`
        to `cod @ mem` back into its last `mem` inputs,
        one time step later.

        Parameters:
            dom : The domain of the result, `arg.dom[:-len(mem)]` by default.
            cod : The codomain of the result, `arg.cod[:-len(mem)]`
                by default.
            mem : The memory type fed back, `arg.cod[-1:]` by default.
            initial_state : Optional state of type `mem` plugged in the input
                memory by :meth:`unroll`, default `None` for an open wire.
            final_effect : Optional effect on `mem` plugged in the output
                memory at the last time step of :meth:`unroll`,
                default `None` for an open wire.

        >>> wait = Diagram.swap(mode, mode).feedback()
        >>> assert wait.dom == wait.cod == mode
        >>> assert wait.mem == mode
        """
        return self.feedback_factory(
            self, dom=dom, cod=cod, mem=mem,
            initial_state=initial_state, final_effect=final_effect)

    def unroll(self, n_steps: int = 1) -> Diagram:
        """
        Unroll the feedback loops of a diagram over `n_steps` time steps,
        by interpreting it as a :class:`discopy.stream.Stream`: every box
        maps to the constant stream and every :class:`Feedback` loop moves
        its memory from the wires to the memory of the stream.

        The `initial_state` of each loop is plugged in its input memory at
        the first time step and its `final_effect` in its output memory at
        the last one; memories with `None` are left as open wires at the end
        of the domain and codomain of the result.

        A feedback loop over a swap acts as a delay line: it outputs its
        `initial_state` at the first time step, then its previous input.

        >>> import numpy as np
        >>> from optyx.core.zw import Create, Select
        >>> wait = Diagram.swap(mode, mode).feedback(
        ...     initial_state=Create(1), final_effect=Select(0))
        >>> assert wait.unroll(2).dom == wait.unroll(2).cod == mode ** 2
        >>> amplitude = (Create(0, 0) >> wait.unroll(2) >> Select(1, 0)\\
        ...     ).to_tensor().eval().array
        >>> assert np.isclose(amplitude, 1)
        """
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1.")
        stream, boundaries = self.stream()
        unrolled = stream.unroll(n_steps - 1).now
        mem = stream.mem.now
        dom = unrolled.dom[:len(unrolled.dom) - len(mem)]
        cod = unrolled.cod[:len(unrolled.cod) - len(mem)]
        initial = self.id(type(mem)()).tensor(*(
            self.id(loop_mem) if state is None else state
            for state, _, loop_mem in boundaries))
        final = self.id(type(mem)()).tensor(*(
            self.id(loop_mem) if effect is None else effect
            for _, effect, loop_mem in boundaries))
        return self.id(dom) @ initial >> unrolled >> self.id(cod) @ final

    def stream(self) -> Stream:
        """
        Convert a diagram to its stream semantics and ordered boundaries.

        Every box maps to the constant stream and every :class:`Feedback`
        loop moves its memory from the wires to the memory of the stream,
        so that ``stream.now`` is the one-step diagram from `dom @ mem` to
        `cod @ mem`, with no feedback boxes left.

        **Why a method and not a property.** On a
        :class:`discopy.stream.Stream`, ``now`` and ``mem`` are fields of a
        value that has already been built, so reading them costs nothing and
        attribute syntax is honest; only ``later`` is a property, and it
        forces the tail. Here there is nothing built yet: every call
        reinterprets the diagram functorially to find the memory of each
        nested loop. A cached ``.stream`` property was considered in #12 and
        rejected, so attribute syntax would promise a field and deliver a
        rebuild. Both :meth:`unroll` and :meth:`optyx.channel.Diagram.now`
        go through this one conversion, which is what keeps nested loops to a
        single definition of memory order.

        Boundary states and effects are not applied to ``stream.now``; they
        are returned immutably, left to right with an outer loop before
        nested loops, for :meth:`unroll` to plug at the first and last time
        steps.

        >>> from optyx.core.zw import Create
        >>> wait = Diagram.swap(mode, mode).feedback(initial_state=Create(1))
        >>> semantics = wait.stream()
        >>> assert semantics.stream.now == Diagram.swap(mode, mode)
        >>> assert semantics.boundaries == (
        ...     FeedbackBoundary(Create(1), None, mode),)
        """
        stream_factory = monoidal_stream.Stream[self.factory]
        ty_factory, boundaries = stream_factory.ob, []

        def ar_map(box):
            if not isinstance(box, self.feedback_factory):
                return box
            boundaries.append(FeedbackBoundary(
                box.initial_state, box.final_effect, box.mem))
            inner = functor(box.arg)
            return stream_factory(
                inner.now, dom=ty_factory(box.dom), cod=ty_factory(box.cod),
                mem=ty_factory(box.mem) @ inner.mem)

        functor = monoidal.Functor(
            ob_map=lambda x: x, ar_map=ar_map,
            dom=self.factory, cod=stream_factory)
        return Stream(functor(self), tuple(boundaries))

    # pylint: disable=too-many-locals
    def to_tensor(
        self, input_dims: list = None
    ) -> tensor.Diagram:
        """Returns a :class:`tensor.Diagram` for evaluation"""
        from optyx.core import zw
        from optyx.utils.misc import is_identity

        if input_dims is None:
            input_dims = [2 for _ in range(len(self.dom))]
        else:
            assert len(self.dom) == len(input_dims), (
                "Input dims length does not match number of input wires"
            )
        layer_dims = input_dims

        if is_identity(self):
            return tensor.Diagram.id(Dim(*[int(i) for i in layer_dims]))

        number_of_input_layer_wires = len(self.dom)
        prev_layers: List[Tuple[int, Box]] = []
        for i, (box, left_offset) in enumerate(zip(self.boxes, self.offsets)):
            right_offset = calculate_right_offset(
                number_of_input_layer_wires, left_offset, len(box.dom)
            )

            max_dim = get_max_dim_for_box(
                left_offset,
                box,
                right_offset,
                input_dims,
                prev_layers
            )
            dims_in = layer_dims[left_offset:left_offset + len(box.dom)]
            dims_out = [max_dim if i > max_dim else i
                        for i in box.determine_output_dimensions(dims_in)]

            prev_layers += [
                (left_offset, replacement_box) for replacement_box in
                box.photon_number_transform(dims_in, dims_out)
            ]

            left = Dim()
            if left_offset > 0:
                left = Dim(*[int(i) for i in layer_dims[0:left_offset]])
            right = Dim()
            if left_offset + len(box.dom) < number_of_input_layer_wires:
                right = Dim(
                    *[int(i) for i in layer_dims[left_offset + len(box.dom):
                                                 number_of_input_layer_wires]]
                )

            number_of_input_layer_wires += -len(box.dom) + len(box.cod)
            cod_layer_dims = (
                layer_dims[0:left_offset]
                + dims_out
                + layer_dims[left_offset + len(box.dom):]
            )
            diagram_ = left @ box.truncation(dims_in, dims_out) @ right

            if i == 0:
                diagram = diagram_
            else:
                diagram = diagram >> diagram_

            layer_dims = cod_layer_dims

        zboxes = tensor.Id(Dim(1))

        # pylint: disable=invalid-name
        for c in diagram.cod:
            zboxes @= zw.ZBox(1, 1, lambda i: 1).truncation(
                input_dims=[int(c.inside[0])], output_dims=[int(c.inside[0])]
            )
        diagram >>= zboxes
        return diagram

    @classmethod
    def from_bosonic_operator(
        cls,
        n_modes,
        operators,
        scalar=1
    ):  # pragma: no cover
        """Create a :class:`zw` diagram from a bosonic operator."""
        # pylint: disable=import-outside-toplevel
        from optyx.core import zw

        # pylint: disable=invalid-name
        d = cls.id(Mode(n_modes))
        annil = zw.Split(2) >> zw.Select(1) @ zw.Id(1)
        create = annil.dagger()
        for idx, dagger in operators:
            if not 0 <= idx < n_modes:
                raise ValueError(f"Index {idx} out of bounds.")
            box = create if dagger else annil
            d = d >> zw.Id(idx) @ box @ zw.Id(n_modes - idx - 1)

        if scalar != 1:
            # pylint: disable=invalid-name
            d = Scalar(scalar) @ d
        return d

    def to_pyzx(self):
        # pylint: disable=import-outside-toplevel
        from optyx.core import zx

        try:
            zx_diagram = zx.ZXDiagram(
                dom=self.dom,
                cod=self.cod,
                inside=self.inside,
            )
        except TypeError:
            raise NotImplementedError(
                "Conversion to PyZX is not implemented for this diagram."
            )
        return zx_diagram.to_pyzx()

    # pylint: disable=invalid-name
    def inflate(self, d):
        r"""
        Translates from an indistinguishable setting
        to a distinguishable one. For a map on :math:`F(\mathbb{C})`,
        obtain a map on :math:`F(\mathbb{C})^{\widetilde{\otimes} d}`.
        """
        assert isinstance(d, int), "Dimension must be an integer"
        assert d > 0, "Dimension must be positive"

        # pylint: disable=invalid-name
        def ob(x):
            return Ty.tensor(
                *(o**d if o.inside[0].name == "mode" else o for o in x)
            )

        return symmetric.Functor(
            ob_map=ob,
            ar_map=lambda f: f.inflate(d),
            cod=Diagram,
            dom=Diagram,
        )(self)


class Box(frobenius.Box, Diagram):
    """A box in an optyx diagram"""

    __ambiguous_inheritance__ = (frobenius.Box,)

    def __init__(self, name, dom, cod, array=None, **params):
        self._array = array
        super().__init__(name, dom, cod, **params)
        self.photon_preservation_behaviour = PhotonNumberPreservation.NON_LO

    def photon_number_transform(
        self, dims_in: list[int], dims_out: list[int]
    ) -> Box:
        if (
            self.photon_preservation_behaviour ==
            PhotonNumberPreservation.LO or
            self.photon_preservation_behaviour ==
            PhotonNumberPreservation.QUBIT
        ):
            return [self]
        elif self.photon_preservation_behaviour == \
                PhotonNumberPreservation.NON_LO:
            from optyx.core.zw import Create, Select

            return_list = []
            if len(dims_in) > 0:
                return_list.append(
                    (
                        Select(*[int(i) for i in np.array(dims_in)-1])
                    )
                )
            if len(dims_out) > 0:
                return_list.append(
                    (
                        Create(*[int(i) for i in np.array(dims_out)-1])
                    )
                )
            return return_list
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement "
                "photon_number_transform method"
            )

    @classmethod
    def get_perm(self, n, d):
        return sorted(sorted(list(range(n))), key=lambda i: i % d)

    def inflate(self, d):
        r"""
        Translates from an indistinguishable setting
        to a distinguishable one. For a map on :math:`F(\mathbb{C})`,
        obtain a map on :math:`F(\mathbb{C})^{\widetilde{\otimes} d}`.
        """

        return (
            Diagram.permutation(self.get_perm(len(self.dom)*d, d),
                                self.dom**d) >>
            self**d >>
            Diagram.permutation(self.get_perm(len(self.cod)*d, d),
                                self.cod**d).dagger()
        )

    def conjugate(self) -> Box:
        """Conjugate the box.
        Inheriting boxes should implement this method.
        Otherwise it is defined by the array."""
        if self._array is not None:
            return type(self)(
                self.name + ".dagger()",
                dom=self.cod,
                cod=self.dom,
                array=self._array.conjugate(),
            )
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support conjugation"
        )

    def dagger(self) -> Box:
        """Return the dagger of the box.
        Inheriting boxes should implement this method.
        Otherwise it is defined by the array."""
        if self._array is not None:
            return type(self)(
                self.name + ".dagger()",
                dom=self.cod,
                cod=self.dom,
                array=self._array.T.conjugate(),
            )
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support dagger"
        )

    def truncation(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ) -> tensor.Box:
        """Create a tensor in the semantics of a ZW diagram.
        Inheriting boxes should implement this method.
        Otherwise it is defined by the array."""
        if self._array is not None:
            return tensor.Box(
                self.name,
                dom=tensor.Dim(2) ** len(self.dom),
                cod=tensor.Dim(2) ** len(self.cod),
                data=self._array,
            )

        if input_dims is None:
            raise ValueError("Input dimensions must be provided.")

        if output_dims is None:
            output_dims = self.determine_output_dimensions(input_dims)

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        shape = (
            *[int(i) for i in output_dims],
            *[int(i) for i in input_dims]
        )
        result_matrix = np.zeros(shape, dtype=complex)

        input_ranges = [range(int(d)) for d in input_dims]
        input_combinations = np.array(
            np.meshgrid(*input_ranges)).T.reshape(-1, len(input_dims)) \
            if input_ranges else np.array([[]])

        non_zero_indices = []
        for inp in map(tuple, input_combinations):
            for trans in self.truncation_specification(
                inp, tuple(output_dims)
            ):
                idx = tuple(trans.out + inp)
                non_zero_indices.append((idx, trans.amp))

        if non_zero_indices:
            configs, coeffs = map(np.array, zip(*non_zero_indices))
            idx = tuple(configs.T)
            result_matrix[idx] = coeffs

        out_dims = Dim(*[int(i) for i in output_dims])
        in_dims = Dim(*[int(i) for i in input_dims])

        if self.is_dagger:
            return tensor.Box(self.name, out_dims, in_dims, result_matrix)
        return tensor.Box(
            self.name, out_dims, in_dims, result_matrix
        ).dagger()

    def determine_output_dimensions(self, input_dims: list[int]) -> list[int]:
        """Determine the output dimensions based on the input dimensions.
        The generators of ZW affect the dimensions
        of the output tensor diagrams.
        Inheriting boxes should implement this method.
        Otherwise it is defined by the array."""
        if self._array is not None:
            return input_dims
        str = "does not support determine_output_dimensions"
        raise NotImplementedError(
            f"{self.__class__.__name__} {str}"
        )

    def to_path(self, dtype=complex):
        """Convert the box to a (Q)path representation.
        It can only be defined for zw boxes which are also part of
        the QPath graphical language
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support to_path"
        )

    def grad(self, var):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support grad"
        )

    def lambdify(self, *symbols, **kwargs):
        # Non-symbolic gates can be returned directly
        return lambda *xs: self

    def subs(self, *args) -> Diagram:
        syms, exprs = zip(*args)
        return self.lambdify(*syms)(*exprs)

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, value):
        """
        A :code:`diagram.Box` can be defined through an array
        which will inform :code:`Box.truncation()`, :code:`Box.dagger()`,
        :code:`Box.conjugate()` and :code:`Box.determine_output_dimensions()`.
        The box need to have fixed dom and cod. The tensor should also have
        fixed dimensions. Usually used for zx boxes
        (Bit) with tensor with dims of 2.
        """
        self._array = value

    def truncation_specification(
        self,
        inp: Tuple[int, ...] = None,
        max_output_dims: Tuple[int, ...] = None
    ) -> Iterable[BasisTransition]:
        pass

    def __pow__(self, n):
        if n == 1:
            return self
        return self @ self ** (n - 1)


class Spider(frobenius.Spider, Box):
    """Abstract spider (dagger-SCFA)

    No amplitudes, or no phases.
    """

    draw_as_spider = True
    color = "green"
    photon_preservation_behaviour = PhotonNumberPreservation.NON_LO

    def conjugate(self):
        return self

    def determine_output_dimensions(self, input_dims: list[int]) -> list[int]:
        if isinstance(self.cod, Bit):
            return [2] * len(self.cod)
        else:
            if len(self.dom) == 0:
                return [2 for _ in range(len(self.cod))]
            return [min(input_dims) for _ in range(len(self.cod))]

    def truncation(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ) -> tensor.Box:
        """
        Create a tensor in the semantics of a ZW/ZX diagram depending
        on the domain and codomain type.

        The truncation is defined as a tensor.Spider with
        EmbeddingTensor layers to fix the dimensions of the spider to
        the lowest dimension of the input wires.
        """
        if isinstance(self.cod, Bit) and isinstance(self.dom, Bit):
            return tensor.Spider(len(self.dom), len(self.cod), Dim(2))

        if input_dims is None:
            raise ValueError("Input dimensions must be provided.")

        spider_dim = min(input_dims) if len(self.dom) > 0 else 2

        # get the embedding layer
        embedding_layer = tensor.Id(1)
        for input_dim in input_dims:
            embedding_layer @= (
                EmbeddingTensor(input_dim, spider_dim)
                if input_dim > spider_dim
                else tensor.Id(Dim(int(input_dim)))
            )

        return embedding_layer >> tensor.Spider(
            len(self.dom), len(self.cod), Dim(int(spider_dim))
        )


class Sum(symmetric.Sum, Box):
    """
    Formal sum of optyx diagrams
    """

    __ambiguous_inheritance__ = (symmetric.Sum,)

    def conjugate(self):
        return sum(term.conjugate() for term in self.terms)

    def to_path(self, dtype: type = complex):
        return sum(term.to_path(dtype) for term in self.terms)

    def eval(self, n_photons=0, permanent=None, dtype=complex):
        """Evaluate the sum of diagrams."""
        # we need to implement the proper sums of qpath diagrams
        # this is only a temporary solution, so that the grad tests pass
        if permanent is None:
            # pylint: disable=import-outside-toplevel
            from optyx.core.path import npperm

            permanent = npperm
        return sum(
            term.to_path(dtype).eval(n_photons, permanent)
            for term in self.terms
        )

    def to_tensor(self, input_dims=None):

        terms = [t.to_tensor(input_dims) for t in self]
        cods = [list(t.cod.inside) for t in terms]

        # figure out the max dims for each idx and set it for all the terms
        max_dims = [
            max(c[i] if len(c) > 0 else 0 for c in cods)
            for i in range(len(cods[0]))
        ]

        # modify the diagrams for all the terms
        # add an embedding layer for each wire to fix the cods
        for i, term in enumerate(terms):
            embedding_layer = tensor.Id(1)
            for wire, d in enumerate(term.cod):
                embedding_layer = embedding_layer @ EmbeddingTensor(
                    d.inside[0], max_dims[wire]
                )
            terms[i] = terms[i] >> embedding_layer
            terms[i].cod = Dim(*max_dims)

        # assemble the diagram
        for i, term in enumerate(terms):
            if i == 0:
                diagram = term
            else:
                diagram += term
        return diagram

    def grad(self, var, **params):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        return sum(term.grad(var, **params) for term in self.terms)


class Swap(frobenius.Swap, Box):
    """Swap in optyx diagram"""

    photon_preservation_behaviour = PhotonNumberPreservation.LO

    def conjugate(self):
        return self

    def to_path(self, dtype: type = complex):
        # pylint: disable=import-outside-toplevel
        from optyx.core.path import Matrix

        return Matrix([0, 1, 1, 0], 2, 2)

    def determine_output_dimensions(self, input_dims: list[int]) -> list[int]:
        """Determine the output dimensions based on the input dimensions."""
        return input_dims[::-1]

    def truncation(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ) -> tensor.Box:
        return tensor.Swap(Dim(int(input_dims[0])), Dim(int(input_dims[1])))


class Scalar(Box):
    """
    Scalar in a diagram

    Example
    -------
    >>> from optyx.core.path import Matrix
    >>> from optyx.core.zw import Create, Select
    >>> from optyx.photonic import BS
    >>> assert Scalar(0.45).to_path() == Matrix(
    ...     [], dom=0, cod=0,
    ...     creations=(), selections=(),
    ...     normalisation=1, scalar=0.45)
    >>> s = Scalar(- 1j * 2 ** (1/2)) @ Create(1, 1) >> \\
    ...     BS.get_kraus() >> Select(2, 0)
    >>> assert np.isclose(s.to_path().eval().array[0], 1)
    """

    def __init__(self, scalar: complex | Symbol):
        if not isinstance(scalar, (Symbol, Mul)):
            self.scalar = complex(scalar)
        else:
            self.scalar = scalar
        super().__init__(
            name="scalar", dom=Mode(0), cod=Mode(0), data=self.scalar
        )
        self.photon_preservation_behaviour = PhotonNumberPreservation.LO

    def conjugate(self):
        return Scalar(self.scalar.conjugate())

    def __str__(self):
        return f"scalar({format_number(self.data)})"

    def to_path(self, dtype: type = complex):
        # pylint: disable=import-outside-toplevel
        from optyx.core.path import Matrix

        return Matrix[dtype]([], 0, 0, scalar=self.scalar)

    def dagger(self) -> Diagram:
        return Scalar(self.scalar.conjugate())

    def subs(self, *args):
        data = rsubs(self.scalar, *args)
        return Scalar(data)

    def grad(self, var, **params):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return Sum((), self.dom, self.cod)
        return Scalar(self.scalar.diff(var))

    def lambdify(self, *symbols, **kwargs):
        # pylint: disable=import-outside-toplevel
        from sympy import lambdify

        return lambda *xs: type(self)(
            lambdify(symbols, self.scalar, **kwargs)(*xs)
        )

    def truncation(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ) -> tensor.Box:
        return tensor.Box(self.name, Dim(1), Dim(1), [self.scalar])

    def determine_output_dimensions(
        self, input_dims: list[int] = None
    ) -> list[int]:
        """Determine the output dimensions"""
        return []


class DualRail(Box):
    """
    A map from :code:`Bit` to :code:`Mode` using the dual rail encoding.
    """

    def __init__(self,
                 is_dagger=False,
                 internal_state=None):
        dom = Mode(2) if is_dagger else Bit(1)
        cod = Bit(1) if is_dagger else Mode(2)
        super().__init__("2R", dom, cod)
        self.internal_state = internal_state
        self.is_dagger = is_dagger
        self.photon_preservation_behaviour = \
            PhotonNumberPreservation.CUSTOM

    def photon_number_transform(self, dims_in, dims_out):
        from optyx.core.zw import Create
        from optyx.core.zx import Z
        from copy import deepcopy

        prev_layers = []
        prev_layers.append(Z(1, 0) if not self.is_dagger else Z(0, 1))
        prev_layers.append(Create(1) if not self.is_dagger else Z(1, 0))
        box = deepcopy(self)
        if not self.is_dagger:
            box.dom = Mode(1)
        prev_layers.append(box)
        return prev_layers if not self.is_dagger else prev_layers[::-1]

    def conjugate(self):
        return self

    def truncation_specification(
        self,
        inp: Tuple[int, ...] = None,
        max_output_dims: Tuple[int, ...] = None
    ) -> Iterable[BasisTransition]:
        out = (1, 0) if inp[0] == 0 else (0, 1)
        yield BasisTransition(
            out=out,
            amp=1.0
        )

    def determine_output_dimensions(self,
                                    input_dims: list[int]) -> list[int]:
        """Determine the output dimensions"""
        if self.is_dagger:
            return [2]
        return [2, 2]

    def inflate(self, d):
        # pylint: disable=import-outside-toplevel
        from optyx.core.zw import W, Endo

        assert self.internal_state is not None, \
            "Internal state must be provided"
        assert len(self.internal_state) == d, \
            "Internal state must be of len d"

        dgrm = DualRail() >> Diagram.tensor(
            *[
                (W(d) >>
                 Diagram.tensor(*[Endo(d_i) for d_i in self.internal_state]))
                for _ in range(2)
            ]
        )
        if self.is_dagger:
            return dgrm.dagger()
        return dgrm

    def dagger(self) -> Diagram:
        return DualRail(not self.is_dagger,
                        internal_state=self.internal_state)


class PhotonThresholdDetector(Box):
    """
    Photon-number non-resolving detector from mode to bit.
    Detects whether one or more photons are present.
    """

    def __init__(self, is_dagger=False):
        if is_dagger:
            super().__init__("PTD dagger", Bit(1), Mode(1))
        else:
            super().__init__("PTD", Mode(1), Bit(1))
        self.is_dagger = is_dagger
        self.photon_preservation_behaviour = \
            PhotonNumberPreservation.CUSTOM

    def photon_number_transform(self, dims_in, dims_out):
        from optyx.core.zw import Create, Select
        from optyx.core.zx import Z

        prev_layers = []
        if self.is_dagger:
            prev_layers.append(Z(1, 0))
            prev_layers.append(Create(dims_out[0]-1))
        else:
            prev_layers.append(Select(0))
            prev_layers.append(Z(0, 1))
        return prev_layers

    def truncation_specification(
        self,
        inp: Tuple[int, ...] = None,
        max_output_dims: Tuple[int, ...] = None
    ) -> Iterable[BasisTransition]:
        yield BasisTransition(
            out=(0,) if inp[0] == 0 else (1,),
            amp=1.0
        )

    def determine_output_dimensions(self, input_dims):
        if self.is_dagger:
            return [MAX_DIM] * len(input_dims)
        return [2] * len(input_dims)

    def conjugate(self):
        return self

    def dagger(self):
        return PhotonThresholdDetector(not self.is_dagger)

    def inflate(self, d):
        # pylint: disable=import-outside-toplevel
        from optyx.core.zw import Add
        dgrm = Add(d) >> PhotonThresholdDetector()
        if self.is_dagger:
            return dgrm.dagger()
        return dgrm


class EmbeddingTensor(tensor.Box):
    """
    Embedding tensor for fixing the dimensions of the output tensor.
    """

    def __init__(self, input_dim: int, output_dim: int):

        embedding_array = np.zeros((output_dim, input_dim), dtype=complex)

        if input_dim < output_dim:
            embedding_array[:input_dim, :input_dim] = np.eye(input_dim)
        else:
            embedding_array[:output_dim, :output_dim] = np.eye(output_dim)

        super().__init__(
            "Embedding",
            Dim(int(input_dim)),
            Dim(int(output_dim)),
            embedding_array.T,
        )

    def conjugate(self):
        return self


def dual_rail(n, internal_states=None):
    """
    Encode n qubits into 2n modes via the dual-rail encoding.
    """

    if internal_states is None:
        d = DualRail()
        for i in range(n - 1):
            d @= DualRail()
    else:
        d = DualRail(internal_state=internal_states[0])
        for i, state in enumerate(internal_states[1:]):
            d @= DualRail(internal_state=state)
    return d


def embedding_tensor(n, dim):
    """
    Obtain 2->dim embedding tensors on n wires.
    """
    d = EmbeddingTensor(2, dim)
    for i in range(n - 1):
        d @= EmbeddingTensor(2, dim)
    return d


def truncation_tensor(
    input_dims: List[int], output_dims: List[int]
) -> tensor.Box:

    assert len(input_dims) == len(
        output_dims
    ), "input_dims and output_dims must have the same length"

    tensor = EmbeddingTensor(input_dims[0], output_dims[0])

    for i in zip(input_dims[1:], output_dims[1:]):

        tensor = tensor @ EmbeddingTensor(i[0], i[1])
    return tensor


class Feedback(monoidal.Bubble, Box):
    """
    A feedback loop connecting the last `mem` outputs of `arg`
    back to its last `mem` inputs, one time step later.

    The delay endofunctor is the identity, so `arg` goes from `dom @ mem`
    to `cod @ mem` and the result from `dom` to `cod`. It is drawn as `arg`
    with a feedback loop and composes with any other diagram; its stream
    semantics is given by :meth:`Diagram.unroll`.

    Parameters:
        arg : The diagram from `dom @ mem` to `cod @ mem` to feed back.
        dom : The domain of the result, `arg.dom[:-len(mem)]` by default.
        cod : The codomain of the result, `arg.cod[:-len(mem)]` by default.
        mem : The memory type fed back, `arg.cod[-1:]` by default.
        initial_state : Optional state of type `mem`, see :meth:`unroll`.
        final_effect : Optional effect on `mem`, see :meth:`unroll`.

    Example
    -------
    The feedback of `CNOT >> SWAP` unrolls to a ladder of CNOT gates,
    here with the plus state as initial state and an open output memory:

    >>> from discopy.drawing import Equation
    >>> from optyx.core.zx import Z, X
    >>> cnot = Z(1, 2) @ bit >> bit @ X(2, 1) @ Scalar(2 ** 0.5)
    >>> plus = Scalar(0.5 ** 0.5) @ Z(0, 1)
    >>> ladder = (cnot >> Diagram.swap(bit, bit)).feedback(
    ...     initial_state=plus)
    >>> Equation(ladder, ladder.unroll(3), symbol="$\\mapsto$").draw(
    ...     path="docs/_static/cnot_ladder.svg")

    .. image:: /_static/cnot_ladder.svg
        :align: center
    """
    __ambiguous_inheritance__ = (monoidal.Bubble,)

    def __init__(self, arg, dom=None, cod=None, mem=None,
                 initial_state=None, final_effect=None):
        mem = arg.cod[-1:] if mem is None else mem
        dom = arg.dom[:len(arg.dom) - len(mem)] if dom is None else dom
        cod = arg.cod[:len(arg.cod) - len(mem)] if cod is None else cod
        if (arg.dom, arg.cod) != (dom @ mem, cod @ mem):
            raise AxiomError(
                f"{arg} is not a diagram from {dom @ mem} to {cod @ mem}")
        self.mem = mem
        self.initial_state, self.final_effect = initial_state, final_effect
        monoidal.Bubble.__init__(
            self, arg, dom=dom, cod=cod, method="feedback_operator")
        Box.__init__(self, str(self), dom, cod)

    def __str__(self):
        args = "" if self.mem == self.arg.cod[-1:] else f"mem={self.mem}"
        for attr in ("initial_state", "final_effect"):
            if getattr(self, attr) is not None:
                prefix = ", " if args else ""
                args += f"{prefix}{attr}={getattr(self, attr)}"
        return f"({self.arg}).feedback({args})"

    def __repr__(self):
        args = f"{repr(self.arg)}, mem={repr(self.mem)}"
        for attr in ("initial_state", "final_effect"):
            if getattr(self, attr) is not None:
                args += f", {attr}={repr(getattr(self, attr))}"
        return f"{type(self).__name__}({args})"

    def truncation(self, input_dims=None, output_dims=None):
        """A feedback loop has no tensor or path semantics until unrolled."""
        raise ValueError(
            "The diagram contains a feedback loop "
            "which must be unrolled before evaluation.")

    determine_output_dimensions = truncation
    photon_number_transform = truncation
    to_path = truncation

    def conjugate(self):
        initial_state = None if self.initial_state is None \
            else self.initial_state.conjugate()
        final_effect = None if self.final_effect is None \
            else self.final_effect.conjugate()
        return self.arg.conjugate().feedback(
            dom=self.dom, cod=self.cod, mem=self.mem,
            initial_state=initial_state, final_effect=final_effect)

    def inflate(self, d):
        mem = Ty.tensor(
            *(o ** d if o.inside[0].name == "mode" else o for o in self.mem))
        initial_state = None if self.initial_state is None \
            else self.initial_state.inflate(d)
        final_effect = None if self.final_effect is None \
            else self.final_effect.inflate(d)
        return self.arg.inflate(d).feedback(
            mem=mem, initial_state=initial_state, final_effect=final_effect)

    def dagger(self):
        raise NotImplementedError(
            "The dagger of a feedback loop is not defined.")

    def to_drawing(self):
        return self.arg.to_drawing().trace(len(self.mem))


class Functor(frobenius.Functor):
    """
    A hypergraph functor is a compact functor that preserves spiders.

    Parameters:
        ob_map (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod : The codomain of the functor, a :class:`Diagram` subclass.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Feedback):
            plugs = {
                attr: self(getattr(other, attr))
                for attr in ("initial_state", "final_effect")
                if getattr(other, attr) is not None}
            return self(other.arg).feedback(
                dom=self(other.dom), cod=self(other.cod),
                mem=self(other.mem), **plugs)
        return super().__call__(other)


class Hypergraph(hypergraph.Hypergraph):  # pragma: no cover
    functor = Functor


bit = Bit(1)
mode = Mode(1)

Diagram.hypergraph_factory = Hypergraph
Diagram.braid_factory, Diagram.spider_factory = Swap, Spider
Diagram.ob = Ty
Diagram.sum_factory = Sum
Diagram.feedback_factory = Feedback
Diagram.functor_factory = Functor
Id = Diagram.id
