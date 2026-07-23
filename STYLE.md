# Code style guide

**optyx is pure.** Diagram composition should never cause side-effects, only functor application does when the codomain is effectful.

**optyx is deterministic.** Even in their internal representation, data structures should not depend on sources of non-determinism (e.g. hashing).

**optyx is transparent.** `eval(repr(x)) == x` should always be true and `eval(str(x)) == x` should be true assuming the obvious variable naming convention. This `str(x)` should be as close as possible to what a mathematician would write on the board.

**optyx has no secrets.** We avoid using private or semiprivate attributes and let the user see the internals of each data structure. We expose the interface of every subprocedure as methods that can be tested and reused.

**optyx cares about naming.** Classes and methods should have short descriptive names, when possible the names correspond to well-known mathematical definitions.

**optyx speaks for itself.** The code should be clear enough that it doesn't need comments, only documentation with links to mathematical definitions.

**optyx does not show off.** If there is a simpler way to name or explain something, don't make it sound more complicated.

**optyx never repeats itself.** Identity and composition of diagrams are defined once, in DisCoPy. If there's duplicate code then you're probably working at the wrong level of abstraction.

**optyx aims at never nesting.** We believe if your code goes beyond three levels deep then you're probably working at the wrong level of abstraction.

**optyx has no TODOs.** Never commit a file or directory named `TODO` to main. Never leave `TODO` comments in the code: open an issue instead.
