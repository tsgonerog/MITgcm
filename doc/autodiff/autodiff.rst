.. _chap_autodiff:

Automatic Differentiation
*************************

Author: Patrick Heimbach

*Automatic differentiation* (AD), also referred to as algorithmic (or,
more loosely, computational) differentiation, involves automatically
deriving code to calculate partial derivatives from an existing fully
non-linear prognostic code (see Griewank and Walther, 2008 :cite:`griewank:08`).
A software
tool is used that parses and transforms source files according to a set
of linguistic and mathematical rules. AD tools are like source-to-source
translators in that they parse a program code as input and produce a new
program code as output (we restrict our discussion to source-to-source
tools, ignoring operator-overloading tools). However, unlike a pure
source-to-source translation, the output program represents a new
algorithm, such as the evaluation of the Jacobian, the Hessian, or
higher derivative operators. In principle, a variety of derived
algorithms can be generated automatically in this way.

MITgcm has been adapted for use with the Tangent linear and Adjoint
Model Compiler (TAMC) and its successor TAF (Transformation of
Algorithms in Fortran), developed by Ralf Giering
(Giering and Kaminski, 1998 :cite:`giering:98`, Giering, 2000
:cite:`giering:00`). The
first application of the adjoint of MITgcm for sensitivity studies was
published by Marotzke et al. (1999) :cite:`maro-eta:99`.
Stammer et al. (1997, 2002) :cite:`stammer:97` :cite:`stammer:02` use MITgcm
and its adjoint for ocean state estimation studies. In the following we shall
refer to TAMC and TAF synonymously, except were explicitly stated otherwise.

As of mid-2007 an open-source AD tool, called OpenAD (see Naumann, 2006
:cite:`naumann:06` and Utke et al., 2008 :cite:`utke:08`) was made available
to generate adjoint code of MITgcm. The support for OpenAD ended in July 2026
as OpenAD was no longer maintained.
By that time the MITgcm interface with open-source AD tool Tapenade was fully
operational (see :numref:`ad_tapenade`) and allows to generate fairly efficient
adjoint and tangent-linear code of MITgcm (Gaikwad et al., 2024 :cite:`gaikwad:24`).

The AD tool exploits the chain rule for computing the first derivative
of a function with respect to a set of input variables. Treating a given
forward code as a composition of operations – each line representing a
compositional element, the chain rule is rigorously applied to the code,
line by line. The resulting tangent linear or adjoint code, then, may be
thought of as the composition in forward or reverse order, respectively,
of the Jacobian matrices of the forward code’s compositional elements.

This chapter is organized as follows. :numref:`sec_autodiff_basic_algebra`
introduces the tangent linear and the adjoint model and the storing of the
model trajectory that the adjoint model requires.
:numref:`sec_ad_tlm_and_adm` describes what is common to all AD tools in
MITgcm: the structure of a tangent linear or adjoint run, the cost function,
the control variables, the code that is given to the AD tool, and the
output. :numref:`ad_gradient_check` describes how a gradient is verified
against finite differences. What is specific to an AD tool, such as its
build procedure and the way in which it is told about hand-written
derivative code, is described in :numref:`ad_taf` for TAF and in
:numref:`ad_tapenade` for Tapenade. :numref:`ad_openad` is kept as a record
of the former OpenAD interface.

.. _sec_autodiff_basic_algebra:

Some basic algebra
==================

Let :math:`\cal{M}` be a general nonlinear, model, i.e., a mapping from
the :math:`m`-dimensional space :math:`U \subset \mathbb{R}^m` of input
variables :math:`\vec{u}=(u_1,\ldots,u_m)` (model parameters, initial
conditions, boundary conditions such as forcing functions) to the
:math:`n`-dimensional space :math:`V \subset \mathbb{R}^n` of model output
variable :math:`\vec{v}=(v_1,\ldots,v_n)` (model state, model
diagnostics, objective function, ...) under consideration:

.. math::
   \begin{aligned}
   {\cal M} \, : & \, U \,\, \longrightarrow \, V \\
   ~      & \, \vec{u} \,\, \longmapsto \, \vec{v} \, = \,
   {\cal M}(\vec{u})\end{aligned}
   :label: fulloperator

The vectors :math:`\vec{u} \in U` and :math:`\vec{v} \in V` may be
represented with respect to some given basis vectors
:math:`{\rm span} (U) = \{ {\vec{e}_i} \}_{i = 1, \ldots , m}` and
:math:`{\rm span} (V) = \{ {\vec{f}_j} \}_{j = 1, \ldots , n}` as

.. math::
   \vec{u} \, = \, \sum_{i=1}^{m} u_i \, {\vec{e}_i},
   \qquad
   \vec{v} \, = \, \sum_{j=1}^{n} v_j \, {\vec{f}_j}

Two routes may be followed to determine the sensitivity of the output
variable :math:`\vec{v}` to its input :math:`\vec{u}`.

Forward or direct sensitivity
-----------------------------

Consider a perturbation to the input variables :math:`\delta \vec{u}`
(typically a single component
:math:`\delta \vec{u} = \delta u_{i} \, {\vec{e}_{i}}`). Their effect on
the output may be obtained via the linear approximation of the model
:math:`{\cal M}` in terms of its Jacobian matrix :math:`M`, evaluated
in the point :math:`u^{(0)}` according to

.. math::
   \delta \vec{v} \, = \, M |_{\vec{u}^{(0)}} \, \delta \vec{u}
   :label: tangent_linear

with resulting output perturbation :math:`\delta \vec{v}`. In
components
:math:`M_{j i} \, = \, \partial {\cal M}_{j} / \partial u_{i}`, it
reads

.. math::
   \delta v_{j} \, = \, \sum_{i}
   \left. \frac{\partial {\cal M}_{j}}{\partial u_{i}} \right|_{u^{(0)}} \,
   \delta u_{i}
   :label: jacobi_matrix

:eq:`tangent_linear` is the tangent linear model (TLM). In contrast
to the full nonlinear model :math:`{\cal M}`, the operator :math:`M`
is just a matrix which can readily be used to find the forward
sensitivity of :math:`\vec{v}` to perturbations in :math:`u`, but if
there are very many input variables :math:`(\gg O(10^{6})` for
large-scale oceanographic application), it quickly becomes prohibitive
to proceed directly as in :eq:`tangent_linear`, if the impact of each
component :math:`{\bf e_{i}}` is to be assessed.

Reverse or adjoint sensitivity
------------------------------

Let us consider the special case of a scalar objective function
:math:`{\cal J}(\vec{v})` of the model output (e.g., the total meridional
heat transport, the total uptake of CO\ :sub:`2` in the Southern Ocean
over a time interval, or a measure of some model-to-data misfit)

.. math::
   \begin{aligned}
   \begin{array}{cccccc}
   {\cal J}  \, : &  U &
   \longrightarrow & V &
   \longrightarrow & \mathbb{R} \\
   ~       &  \vec{u} & \longmapsto     & \vec{v}={\cal M}(\vec{u}) &
   \longmapsto     & {\cal J}(\vec{u}) = {\cal J}({\cal M}(\vec{u}))
   \end{array}\end{aligned}
   :label: compo

The perturbation of :math:`{\cal J}` around a fixed point
:math:`{\cal J}_0`,

.. math:: {\cal J} \, = \, {\cal J}_0 \, + \, \delta {\cal J}

can be expressed in both bases of :math:`\vec{u}` and
:math:`\vec{v}` with respect to their corresponding inner product
:math:`\left\langle \,\, , \,\, \right\rangle`

.. math::
   \begin{aligned}
   {\cal J} & = \,
   {\cal J} |_{\vec{u}^{(0)}} \, + \,
   \left\langle \, \nabla _{u}{\cal J}^T |_{\vec{u}^{(0)}} \, , \, \delta \vec{u} \, \right\rangle
   \, + \, O(\delta \vec{u}^2) \\
   ~ & = \,
   {\cal J} |_{\vec{v}^{(0)}} \, + \,
   \left\langle \, \nabla _{v}{\cal J}^T |_{\vec{v}^{(0)}} \, , \, \delta \vec{v} \, \right\rangle
   \, + \, O(\delta \vec{v}^2)
   \end{aligned}
   :label: deljidentity

(note, that the gradient :math:`\nabla f` is a co-vector, therefore
its transpose is required in the above inner product). Then, using the
representation of :math:`\delta {\cal J} =
\left\langle \, \nabla _{v}{\cal J}^T \, , \, \delta \vec{v} \, \right\rangle`,
the definition of an adjoint operator :math:`A^{\ast}` of a given
operator :math:`A`,

.. math::
   \left\langle \, A^{\ast} \vec{x} \, , \, \vec{y} \, \right\rangle =
   \left\langle \, \vec{x} \, , \,  A \vec{y} \, \right\rangle

which for finite-dimensional vector spaces is just the transpose of
:math:`A`,

.. math:: A^{\ast} \, = \, A^T

and from :eq:`tangent_linear`, :eq:`deljidentity`, we note that
(omitting :math:`|`\ ’s):

.. math::
   \delta {\cal J}
   \, = \,
   \left\langle \, \nabla _{v}{\cal J}^T \, , \, \delta \vec{v} \, \right\rangle
   \, = \,
   \left\langle \, \nabla _{v}{\cal J}^T \, , \, M \, \delta \vec{u} \, \right\rangle
   \, = \,
   \left\langle \, M^T \, \nabla _{v}{\cal J}^T \, , \,
   \delta \vec{u} \, \right\rangle
   :label: inner

With the identity :eq:`deljidentity`, we then find that the gradient
:math:`\nabla _{u}{\cal J}` can be readily inferred by invoking the
adjoint :math:`M^{\ast }` of the tangent linear model :math:`M`

.. math::
   \begin{aligned}
   \nabla _{u}{\cal J}^T |_{\vec{u}} &
   = \, M^T |_{\vec{u}} \cdot \nabla _{v}{\cal J}^T |_{\vec{v}}  \\
   ~ & = \, M^T |_{\vec{u}} \cdot \delta \vec{v}^{\ast} \\
   ~ & = \, \delta \vec{u}^{\ast}
   \end{aligned}
   :label: adjoint

:eq:`adjoint` is the adjoint model (ADM), in which :math:`M^T` is the
adjoint (here, the transpose) of the tangent linear operator :math:`M`,
:math:`\,\delta \vec{v}^{\ast}` the adjoint variable of the model state
:math:`\vec{v}`, and :math:`\delta \vec{u}^{\ast}` the adjoint
variable of the control variable :math:`\vec{u}`.

The reverse nature of the adjoint calculation can be readily seen as
follows. Consider a model integration which consists of
:math:`\Lambda` consecutive operations
:math:`{\cal M}_{\Lambda} (  {\cal M}_{\Lambda-1} ( ...... ( {\cal M}_{\lambda} (......
( {\cal M}_{1} ( {\cal M}_{0}(\vec{u}) ))))`, where the
:math:`{\cal M}`\ ’s could be the elementary steps, i.e., single lines in
the code of the model, or successive time steps of the model
integration, starting at step 0 and moving up to step :math:`\Lambda`,
with intermediate
:math:`{\cal M}_{\lambda} (\vec{u}) = \vec{v}^{(\lambda+1)}` and final
:math:`{\cal M}_{\Lambda} (\vec{u}) = \vec{v}^{(\Lambda+1)} = \vec{v}`.
Let :math:`{\cal J}` be a cost function which explicitly depends on the
final state :math:`\vec{v}` only (this restriction is for clarity
reasons only). :math:`{\cal J}(u)` may be decomposed according to:

.. math::
   {\cal J}({\cal M}(\vec{u})) \, = \,
   {\cal J} ( {\cal M}_{\Lambda} (  {\cal M}_{\Lambda-1} (
   ...... ( {\cal M}_{\lambda} (......
   ( {\cal M}_{1} ( {\cal M}_{0}(\vec{u}) )))))
   :label: compos

Then, according to the chain rule, the forward calculation reads, in
terms of the Jacobi matrices (we’ve omitted the :math:`|`\ ’s which,
nevertheless are important to the aspect of *tangent* linearity; note
also that by definition
:math:`\langle \, \nabla _{v}{\cal J}^T \, , \, \delta \vec{v} \, \rangle
= \nabla_v {\cal J} \cdot \delta \vec{v}` )

.. math::
   \begin{aligned}
   \nabla_v {\cal J} (M(\delta \vec{u})) & = \,
   \nabla_v {\cal J} \cdot M_{\Lambda}
   \cdot ...... \cdot M_{\lambda} \cdot ...... \cdot
   M_{1} \cdot M_{0} \cdot \delta \vec{u} \\
   ~ & = \, \nabla_v {\cal J} \cdot \delta \vec{v} \\
   \end{aligned}
   :label: forward

whereas in reverse mode we have

.. math::
   \boxed{
   \begin{aligned}
   M^T ( \nabla_v {\cal J}^T) & = \,
   M_{0}^T \cdot M_{1}^T
   \cdot ...... \cdot M_{\lambda}^T \cdot ...... \cdot
   M_{\Lambda}^T \cdot \nabla_v {\cal J}^T \\
   ~ & = \, M_{0}^T \cdot M_{1}^T
   \cdot ...... \cdot
   \nabla_{v^{(\lambda)}} {\cal J}^T \\
   ~ & = \, \nabla_u {\cal J}^T
   \end{aligned}}
   :label: reverse

clearly expressing the reverse nature of the calculation.
:eq:`reverse` is at the heart of automatic adjoint compilers. If the
intermediate steps :math:`\lambda` in :eq:`compos` – :eq:`reverse`
represent the model state (forward or adjoint) at each intermediate time
step as noted above, then correspondingly,
:math:`M^T (\delta \vec{v}^{(\lambda) \, \ast}) =
\delta \vec{v}^{(\lambda-1) \, \ast}` for the adjoint variables. It
thus becomes evident that the adjoint calculation also yields the
adjoint of each model state component :math:`\vec{v}^{(\lambda)}` at
each intermediate step :math:`\lambda`, namely

.. math::
   \boxed{
   \begin{aligned}
   \nabla_{v^{(\lambda)}} {\cal J}^T |_{\vec{v}^{(\lambda)}}
   & = \,
   M_{\lambda}^T |_{\vec{v}^{(\lambda)}} \cdot ...... \cdot
   M_{\Lambda}^T |_{\vec{v}^{(\lambda)}} \cdot \delta \vec{v}^{\ast} \\
   ~ & = \, \delta \vec{v}^{(\lambda) \, \ast}
   \end{aligned}}

in close analogy to :eq:`adjoint` we note in passing that the
:math:`\delta \vec{v}^{(\lambda) \, \ast}` are the Lagrange multipliers
of the model equations which determine :math:`\vec{v}^{(\lambda)}`.

In components, :eq:`adjoint` reads as follows. Let

.. math::
   \begin{array}{rclcrcl}
   \delta \vec{u} & = &
   \left( \delta u_1,\ldots, \delta u_m \right)^T , & \qquad &
   \delta \vec{u}^{\ast} \,\, = \,\, \nabla_u {\cal J}^T & = &
   \left(
   \frac{\partial {\cal J}}{\partial u_1},\ldots,
   \frac{\partial {\cal J}}{\partial u_m}
   \right)^T \\
   \delta \vec{v} & = &
   \left( \delta v_1,\ldots, \delta u_n \right)^T , & \qquad &
   \delta \vec{v}^{\ast} \,\, = \,\, \nabla_v {\cal J}^T & = &
   \left(
   \frac{\partial {\cal J}}{\partial v_1},\ldots,
   \frac{\partial {\cal J}}{\partial v_n}
   \right)^T \\
   \end{array}

denote the perturbations in :math:`\vec{u}` and :math:`\vec{v}`,
respectively, and their adjoint variables; further

.. math::
   M \, = \, \left(
   \begin{array}{ccc}
   \frac{\partial {\cal M}_1}{\partial u_1} & \ldots &
   \frac{\partial {\cal M}_1}{\partial u_m} \\
   \vdots & ~ & \vdots \\
   \frac{\partial {\cal M}_n}{\partial u_1} & \ldots &
   \frac{\partial {\cal M}_n}{\partial u_m} \\
   \end{array}
   \right)

is the Jacobi matrix of :math:`{\cal M}` (an :math:`n \times m`
matrix) such that :math:`\delta \vec{v} = M \cdot \delta \vec{u}`, or

.. math::
   \delta v_{j}
   \, = \, \sum_{i=1}^m M_{ji} \, \delta u_{i}
   \, = \, \sum_{i=1}^m \, \frac{\partial {\cal M}_{j}}{\partial u_{i}}
   \delta u_{i}

Then :eq:`adjoint` takes the form

.. math::
   \delta u_{i}^{\ast}
   \, = \, \sum_{j=1}^n M_{ji} \, \delta v_{j}^{\ast}
   \, = \, \sum_{j=1}^n \, \frac{\partial {\cal M}_{j}}{\partial u_{i}}
   \delta v_{j}^{\ast}

or

.. math::
   \left(
   \begin{array}{c}
   \left. \frac{\partial}{\partial u_1} {\cal J} \right|_{\vec{u}^{(0)}} \\
   \vdots \\
   \left. \frac{\partial}{\partial u_m} {\cal J} \right|_{\vec{u}^{(0)}} \\
   \end{array}
   \right)
   \, = \,
   \left(
   \begin{array}{ccc}
   \left. \frac{\partial {\cal M}_1}{\partial u_1} \right|_{\vec{u}^{(0)}}
   & \ldots &
   \left. \frac{\partial {\cal M}_n}{\partial u_1} \right|_{\vec{u}^{(0)}} \\
   \vdots & ~ & \vdots \\
   \left. \frac{\partial {\cal M}_1}{\partial u_m} \right|_{\vec{u}^{(0)}}
   & \ldots &
   \left. \frac{\partial {\cal M}_n}{\partial u_m} \right|_{\vec{u}^{(0)}} \\
   \end{array}
   \right)
   \cdot
   \left(
   \begin{array}{c}
   \left. \frac{\partial}{\partial v_1} {\cal J} \right|_{\vec{v}} \\
   \vdots \\
   \left. \frac{\partial}{\partial v_n} {\cal J} \right|_{\vec{v}} \\
   \end{array}
   \right)

Furthermore, the adjoint :math:`\delta v^{(\lambda) \, \ast}` of any
intermediate state :math:`v^{(\lambda)}` may be obtained, using the
intermediate Jacobian (an :math:`n_{\lambda+1} \times n_{\lambda}`
matrix)

.. math::
   M_{\lambda} \, = \,
   \left(
   \begin{array}{ccc}
   \frac{\partial ({\cal M}_{\lambda})_1}{\partial v^{(\lambda)}_1}
   & \ldots &
   \frac{\partial ({\cal M}_{\lambda})_1}{\partial v^{(\lambda)}_{n_{\lambda}}} \\
   \vdots & ~ & \vdots \\
   \frac{\partial ({\cal M}_{\lambda})_{n_{\lambda+1}}}{\partial v^{(\lambda)}_1}
   & \ldots &
   \frac{\partial ({\cal M}_{\lambda})_{n_{\lambda+1}}}{\partial v^{(\lambda)}_{n_{\lambda}}} \\
   \end{array}
   \right)

and the shorthand notation for the adjoint variables
:math:`\delta v^{(\lambda) \, \ast}_{j} = \frac{\partial}{\partial v^{(\lambda)}_{j}}
{\cal J}^T`, :math:`j = 1, \ldots , n_{\lambda}`, for intermediate
components, yielding

.. math::
   \begin{aligned}
   \left(
   \begin{array}{c}
   \delta v^{(\lambda) \, \ast}_1 \\
   \vdots \\
   \delta v^{(\lambda) \, \ast}_{n_{\lambda}} \\
   \end{array}
   \right)
   \, = &
   \left(
   \begin{array}{ccc}
   \frac{\partial ({\cal M}_{\lambda})_1}{\partial v^{(\lambda)}_1}
   & \ldots \,\, \ldots &
   \frac{\partial ({\cal M}_{\lambda})_{n_{\lambda+1}}}{\partial v^{(\lambda)}_1} \\
   \vdots & ~ & \vdots \\
   \frac{\partial ({\cal M}_{\lambda})_1}{\partial v^{(\lambda)}_{n_{\lambda}}}
   & \ldots \,\, \ldots  &
   \frac{\partial ({\cal M}_{\lambda})_{n_{\lambda+1}}}{\partial v^{(\lambda)}_{n_{\lambda}}} \\
   \end{array}
   \right)
   \cdot
   %
   \\ ~ & ~
   \\ ~ &
   %
   \left(
   \begin{array}{ccc}
   \frac{\partial ({\cal M}_{\lambda+1})_1}{\partial v^{(\lambda+1)}_1}
   & \ldots &
   \frac{\partial ({\cal M}_{\lambda+1})_{n_{\lambda+2}}}{\partial v^{(\lambda+1)}_1} \\
   \vdots & ~ & \vdots \\
   \vdots & ~ & \vdots \\
   \frac{\partial ({\cal M}_{\lambda+1})_1}{\partial v^{(\lambda+1)}_{n_{\lambda+1}}}
   & \ldots  &
   \frac{\partial ({\cal M}_{\lambda+1})_{n_{\lambda+2}}}{\partial v^{(\lambda+1)}_{n_{\lambda+1}}} \\
   \end{array}
   \right)
   \cdot \, \ldots \, \cdot
   \left(
   \begin{array}{c}
   \delta v^{\ast}_1 \\
   \vdots \\
   \delta v^{\ast}_{n} \\
   \end{array}
   \right)
   \end{aligned}

:eq:`forward` and :eq:`reverse` are perhaps clearest in showing the
advantage of the reverse over the forward mode if the gradient
:math:`\nabla _{u}{\cal J}`, i.e., the sensitivity of the cost function
:math:`{\cal J}` with respect to *all* input variables :math:`u` (or
the sensitivity of the cost function with respect to *all* intermediate
states :math:`\vec{v}^{(\lambda)}`) are sought. In order to be able to
solve for each component of the gradient
:math:`\partial {\cal J} / \partial u_{i}` in :eq:`forward` a forward
calculation has to be performed for each component separately, i.e.,
:math:`\delta \vec{u} = \delta u_{i} {\vec{e}_{i}}` for the
:math:`i`-th forward calculation. Then, :eq:`forward` represents the
projection of :math:`\nabla_u {\cal J}` onto the :math:`i`-th
component. The full gradient is retrieved from the :math:`m` forward
calculations. In contrast, :eq:`reverse` yields the full gradient
:math:`\nabla _{u}{\cal J}` (and all intermediate gradients
:math:`\nabla _{v^{(\lambda)}}{\cal J}`) within a single reverse
calculation.

Note, that if :math:`{\cal J}` is a vector-valued function of
dimension :math:`l > 1`, :eq:`reverse` has to be modified according
to

.. math::
   M^T \left( \nabla_v {\cal J}^T \left(\delta \vec{J}\right) \right)
   \, = \,
   \nabla_u {\cal J}^T \cdot \delta \vec{J}

where now :math:`\delta \vec{J} \in \mathbb{R}^l` is a vector of
dimension :math:`l`. In this case :math:`l` reverse simulations have
to be performed for each :math:`\delta J_{k}, \,\, k = 1, \ldots, l`.
Then, the reverse mode is more efficient as long as :math:`l < n`,
otherwise the forward mode is preferable. Strictly, the reverse mode is
called adjoint mode only for :math:`l = 1`.

A detailed analysis of the underlying numerical operations shows that
the computation of :math:`\nabla _{u}{\cal J}` in this way requires
about two to five times the computation of the cost function. Alternatively,
the gradient vector could be approximated by finite differences,
requiring :math:`m` computations of the perturbed cost function.

To conclude, we give two examples of commonly used types of cost
functions:

Example 1: :math:`{\cal J} = v_{j} (T)`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cost function consists of the :math:`j`-th component of the model
state :math:`\vec{v}` at time :math:`T`. Then
:math:`\nabla_v {\cal J}^T = {\vec{f}_{j}}` is just the :math:`j`-th
unit vector. The :math:`\nabla_u {\cal J}^T` is the projection of
the adjoint operator onto the :math:`j`-th component
:math:`{\bf f_{j}}`,

.. math::
     \nabla_u {\cal J}^T
     \, = \, M^T \cdot \nabla_v {\cal J}^T
     \, = \,  \sum_{i} M^T_{ji} \, {\vec{e}_{i}}

Example 2: :math:`{\cal J} = \langle \, {\cal H}(\vec{v}) - \vec{d} \, , \, {\cal H}(\vec{v}) - \vec{d} \, \rangle`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cost function represents the quadratic model vs. data misfit.
Here, :math:`\vec{d}` is the data vector and :math:`{\cal H}`
represents the operator which maps the model state space onto the data
space. Then, :math:`\nabla_v {\cal J}` takes the form

.. math::
     \begin{aligned}
     \nabla_v {\cal J}^T & = \, 2 \, \, H \cdot
     \left( \, {\cal H}(\vec{v}) - \vec{d} \, \right) \\
     ~          & = \, 2 \sum_{j} \left\{ \sum_k
     \frac{\partial {\cal H}_k}{\partial v_{j}}
     \left( {\cal H}_k (\vec{v}) - d_k \right)
     \right\} \, {\vec{f}_{j}} \\
     \end{aligned}

where :math:`H_{kj} = \partial {\cal H}_k / \partial v_{j}` is the
Jacobi matrix of the data projection operator. Thus, the gradient
:math:`\nabla_u {\cal J}` is given by the adjoint operator, driven
by the model vs. data misfit:

.. math::
    \nabla_u {\cal J}^T \, = \, 2 \, M^T \cdot
     H \cdot \left( {\cal H}(\vec{v}) - \vec{d} \, \right)

.. _sec_autodiff_storage_v_recompute:

Storing vs. recomputation in reverse mode
-----------------------------------------

We note an important aspect of the forward vs. reverse mode calculation.
Because of the local character of the derivative (a derivative is
defined with respect to a point along the trajectory), the intermediate results
of the model trajectory
:math:`\vec{v}^{(\lambda+1)}={\cal M}_{\lambda}(v^{(\lambda)})` may be
required to evaluate the intermediate Jacobian
:math:`M_{\lambda}|_{\vec{v}^{(\lambda)}} \, \delta \vec{v}^{(\lambda)}`.
This is the case for example for nonlinear expressions (momentum advection,
nonlinear equation of state), and state-dependent conditional statements
(parameterization schemes). In the forward mode, the intermediate
results are required in the same order as computed by the full forward
model :math:`{\cal M}`, but in the reverse mode they are required in the
reverse order. Thus, in the reverse mode the trajectory of the forward
model integration :math:`{\cal M}` has to be stored to be available in
the reverse calculation. Alternatively, the complete model state up to
the point of evaluation has to be recomputed whenever its value is
required.

A method to balance the amount of recomputations vs. storage
requirements is called checkpointing (e.g., Griewank, 1992 :cite:`griewank:92`,
Restrepo et al., 1998 :cite:`restrepo:98`). It is depicted in :numref:`checkpointing` for
a 3-level checkpointing (as an example, we give explicit numbers for a
3-day integration with a 1-hourly timestep in square brackets).

 .. figure:: figs/checkpointing.png
    :width: 100%
    :align: center
    :alt: 3-lvl checkpointing schematic figure
    :name: checkpointing

    Schematic view of intermediate dump and restart for 3-level checkpointing.

-  In a first step, the model trajectory is subdivided into
   :math:`{n}^{lev3}` subsections [:math:`{n}^{lev3}`\ =3 1-day
   intervals], with the label :math:`lev3` for this outermost loop. The
   model is then integrated along the full trajectory, and the model
   state stored to disk only at every :math:`k_{i}^{lev3}`-th timestep
   [i.e. 3 times, at :math:`i = 0,1,2` corresponding to
   :math:`k_{i}^{lev3} = 0, 24, 48`]. In addition, the cost function
   is computed, if needed.

-  In a second step each subsection itself is divided into
   :math:`{n}^{lev2}` subsections [:math:`{n}^{lev2}`\ =4 6-hour
   intervals per subsection]. The model picks up at the last outermost
   dumped state :math:`v_{k_{n}^{lev3}}` and is integrated forward in
   time along the last subsection, with the label :math:`lev2` for this
   intermediate loop. The model state is now stored to disk at every
   :math:`k_{i}^{lev2}`-th timestep [i.e. 4 times, at
   :math:`i = 0,1,2,3` corresponding to
   :math:`k_{i}^{lev2} = 48, 54, 60, 66`].

-  Finally, the model picks up at the last intermediate dump state
   :math:`v_{k_{n}^{lev2}}` and is integrated forward in time along
   the last subsection, with the label :math:`lev1` for this
   intermediate loop. Within this sub-subsection only, parts of the
   model state are stored to memory at every timestep [i.e. every hour
   :math:`i=0,...,5` corresponding to
   :math:`k_{i}^{lev1} = 66, 67, \ldots, 71`]. The final state
   :math:`v_n = v_{k_{n}^{lev1}}` is reached and the model state of
   all preceding timesteps along the last innermost subsection are
   available, enabling integration backwards in time along the last
   subsection. The adjoint can thus be computed along this last
   subsection :math:`k_{n}^{lev2}`.

This procedure is repeated consecutively for each previous subsection
:math:`k_{n-1}^{lev2}, \ldots, k_{1}^{lev2}` carrying the adjoint
computation to the initial time of the subsection :math:`k_{n}^{lev3}`.
Then, the procedure is repeated for the previous subsection
:math:`k_{n-1}^{lev3}` carrying the adjoint computation to the initial
time :math:`k_{1}^{lev3}`.

For the full model trajectory of
:math:`n^{lev3} \cdot n^{lev2} \cdot n^{lev1}` timesteps the required
storing of the model state was significantly reduced to
:math:`n^{lev2} + n^{lev3}` to disk and roughly :math:`n^{lev1}` to
memory (i.e., for the 3-day integration with a total of 72 timesteps the
model state was stored 7 times to disk and roughly 6 times to memory).
This saving in memory comes at a cost of a required 3 full forward
integrations of the model (one for each checkpointing level). The
optimal balance of storage vs. recomputation certainly depends on the
computing resources available and may be adjusted by adjusting the
partitioning among the :math:`n^{lev3}, \,\, n^{lev2}, \,\, n^{lev1}`.

.. _sec_ad_tlm_and_adm:

TLM and ADM generation in general
=================================

Authors: Patrick Heimbach and Tanvir Shahriar

In this section we describe in a general fashion the parts of the code
that are relevant for automatic differentiation, whichever AD tool is
used. MITgcm can be differentiated with TAF and with Tapenade. Both tools
are given the same top-level routine, the same dependent variable,
independent variables of the same kind and, apart from a few tool-specific
files, the same source code.
They differ in the names of the code they generate, in the way they are
told about code that they do not differentiate, and in the way the
trajectory of the forward model is stored. These differences are pointed
out below where they matter; the build procedure and the other details of
each tool are described in :numref:`ad_taf` (TAF) and
:numref:`ad_tapenade` (Tapenade).

.. _sec_ad_run_structure:

Structure of a tangent linear or adjoint run
--------------------------------------------

The AD tool differentiates one routine and everything that is called from
it: S/R :filelink:`the_main_loop.F <model/src/the_main_loop.F>`, which
initializes the model state, steps the model forward in time and evaluates
the cost function. The cost function :varlink:`fc` is the dependent
variable and the control variables are the independent variables; they are
described in :numref:`pkg_cost_description` and :numref:`the_ctrl_vars`.
From this top-level routine the tool generates the tangent linear or the
adjoint routine, and the driver routine :filelink:`the_model_main.F
<model/src/the_model_main.F>` calls the generated routine instead of
:filelink:`the_main_loop.F <model/src/the_main_loop.F>`. For an adjoint run
the basic flow is as follows (the generated routines are shown with the
names that TAF gives them; :numref:`tab_ad_naming` lists those of
Tapenade):

::

       the_model_main
       |
       |--- initialise_fixed
       |
       |--- ctrl_unpack
       |
       |--- #ifdef ALLOW_ADJOINT_RUN
       |           |
       |           |--- adthe_main_loop
       |           |    |
       |           |    |--- initialise_varia          - forward sweep
       |           |    |--- do iloop = 1, nTimeSteps
       |           |    |       |--- forward_step
       |           |    |    end do
       |           |    |--- cost_final
       |           |    |
       |           |    |--- adcost_final              - reverse sweep
       |           |    |--- do iloop = nTimeSteps, 1, -1
       |           |    |       |--- adforward_step
       |           |    |    end do
       |           |    |--- adinitialise_varia
       |           |    o
       |           |
       |--- #else
       |           |
       |           |--- the_main_loop
       |           |
       |    #endif
       |
       |--- ctrl_pack
       |
       |--- #ifdef ALLOW_GRDCHK
       |           |
       |           |--- grdchk_main
       |           o
       |    #endif
       o

The generated routine ``adthe_main_loop`` contains both the forward
integration of the full model and the reverse integration of the adjoint
model. In its first half, the *forward sweep*, the model is integrated
forward in time, the cost function is calculated, and the intermediate
results that the second half requires are stored. In its second half, the
*reverse sweep* (or backward sweep), the adjoint variables are propagated
backward in time, from the cost function to the control variables. The
above structure has been strongly simplified to focus on the essentials; in
particular, no checkpointing procedures are shown here (see
:numref:`sec_autodiff_storage_v_recompute` and :numref:`sec_ad_storing`).

Prior to the call of ``adthe_main_loop``, the routine
:filelink:`ctrl_unpack.F <pkg/ctrl/ctrl_unpack.F>` is invoked to unpack the
control vector, if an optimization has provided one. Following the call of
``adthe_main_loop``, the routine :filelink:`ctrl_pack.F
<pkg/ctrl/ctrl_pack.F>` is invoked to pack the gradient vector (cf.
:numref:`the_ctrl_vars` and :numref:`sec_ad_optim`). If gradient checks are
to be performed
(:filelink:`pkg/grdchk` compiled and :varlink:`useGrdchk` ``= .TRUE.`` in
``data.pkg``), the driver routine :filelink:`grdchk_main.F
<pkg/grdchk/grdchk_main.F>` is called after the gradient has been computed
via the adjoint (cf. :numref:`ad_gradient_check`).

In a tangent linear run the tangent linear of :filelink:`the_main_loop.F
<model/src/the_main_loop.F>` takes the place of its adjoint. It integrates
the model and a perturbation of the control variables forward in time
together, so that there is no reverse sweep. With Tapenade the tangent
linear routine is not called at this place but from the gradient check
package (:numref:`ad_gradient_check`).

The two AD tools name the code they generate differently
(:numref:`tab_ad_naming`). These names appear in the generated source
files, in the hand-written derivative code (:numref:`sec_ad_code`) and in
the standard output of a run.

.. table:: Names of the code generated by TAF and by Tapenade.
   :name: tab_ad_naming

   +----------------------------------------------+---------------------+----------------------+
   |                                              | TAF                 | Tapenade             |
   +==============================================+=====================+======================+
   | adjoint of S/R ``the_main_loop``             | ``adthe_main_loop`` | ``the_main_loop_b``  |
   +----------------------------------------------+---------------------+----------------------+
   | tangent linear of S/R ``the_main_loop``      | ``g_the_main_loop`` | ``the_main_loop_d``  |
   +----------------------------------------------+---------------------+----------------------+
   | adjoint of variable ``theta``                | ``adtheta``         | ``thetab``           |
   +----------------------------------------------+---------------------+----------------------+
   | tangent linear of variable ``theta``         | ``g_theta``         | ``thetad``           |
   +----------------------------------------------+---------------------+----------------------+
   | adjoint model executable                     | ``mitgcmuv_ad``     | ``mitgcmuv_tap_adj`` |
   +----------------------------------------------+---------------------+----------------------+
   | tangent linear model executable              | ``mitgcmuv_ftl``    | ``mitgcmuv_tap_tlm`` |
   +----------------------------------------------+---------------------+----------------------+

Which of the AD-related code is compiled is determined by the following CPP
options:

- :varlink:`ALLOW_AUTODIFF` is defined when :filelink:`pkg/autodiff` is
  compiled. It encloses code that is required for automatic differentiation
  with any tool.

- :varlink:`ALLOW_AUTODIFF_TAMC` (file :filelink:`AUTODIFF_OPTIONS.h
  <pkg/autodiff/AUTODIFF_OPTIONS.h>`) encloses code that is required by TAF
  only, mainly its store directives (:numref:`ad_taf_checkpointing`). It
  has to be undefined in a Tapenade setup.

- :varlink:`ALLOW_TAPENADE` is defined when :filelink:`pkg/tapenade` is
  compiled. It encloses code that is required by Tapenade only.

- :varlink:`ALLOW_ADJOINT_RUN` and :varlink:`ALLOW_TANGENTLINEAR_RUN` select
  the adjoint or the tangent linear run in :filelink:`the_model_main.F
  <model/src/the_model_main.F>`. They are not set by the user: the ``make``
  target writes one of them to the header file ``AD_CONFIG.h``
  (:numref:`sec_ad_build`).

.. _sec_ad_general_setup:

General setup
-------------

In order to configure AD-related setups the following packages need to
be enabled:

- :filelink:`pkg/autodiff`, the support code for automatic differentiation:
  the routines that read control variables from file and write their
  gradients (active file routines), hand-written derivative code, and the
  run-time parameters of ``data.autodiff``;

- :filelink:`pkg/ctrl`, the control variables;

- :filelink:`pkg/cost`, the cost function;

- :filelink:`pkg/grdchk`, the gradient check;

- :filelink:`pkg/tapenade`, with Tapenade only: its run-time support and
  the hand-written derivative code in the form that Tapenade calls.

The first four form the package group ``adjoint``. The packages are enabled
by adding them to your experiment-specific configuration file
``packages.conf`` (see :numref:`using_packages`).

The following AD-specific CPP option files need to be customized:

- :filelink:`AUTODIFF_OPTIONS.h <pkg/autodiff/AUTODIFF_OPTIONS.h>` This header
  file collects the CPP options of :filelink:`pkg/autodiff`: the options
  that are specific to TAF and to its checkpointing, and the option
  :varlink:`ALLOW_AUTODIFF_MONITOR` for the output of the adjoint state
  (:numref:`ad_output_dumps_diags`).

- :filelink:`COST_OPTIONS.h <pkg/cost/COST_OPTIONS.h>` In this header file,
  options for different cost functions are set.

- :filelink:`CTRL_OPTIONS.h <pkg/ctrl/CTRL_OPTIONS.h>` In this header file the
  control variables are enabled and options for writing and reading the control
  vector are set; :filelink:`CTRL_SIZE.h <pkg/ctrl/CTRL_SIZE.h>` sets the
  maximum number of control variables.

- :filelink:`tamc.h <pkg/autodiff/tamc.h>`, with TAF only:
  This header configures the splitting of the time stepping loop
  with respect to the 3-level checkpointing (see
  :numref:`ad_taf_checkpointing`).

The run-time parameters are read from the files ``data.autodiff``,
``data.cost``, ``data.ctrl``, ``data.grdchk`` and ``data.optim``.

The verification experiments keep the code and the run-time parameters of
their AD setups in the directories ``code_ad`` and ``input_ad`` for TAF,
and ``code_tap`` and ``input_tap`` for Tapenade; these are a good starting
point for a new setup.

.. _pkg_cost_description:

The cost function (dependent variable)
--------------------------------------

The cost function :math:`{\cal J}` is referred to as the *dependent
variable*. It is a function of the input variables :math:`\vec{u}` via
the composition
:math:`{\cal J}(\vec{u}) \, = \, {\cal J}(M(\vec{u}))`. The input are
referred to as the *independent variables* or *control variables*. All
aspects relevant to the treatment of the cost function
:math:`{\cal J}` (parameter setting, initialization, accumulation,
final evaluation), are controlled by the package :filelink:`pkg/cost`. The aspects
relevant to the treatment of the independent variables are controlled by
the package :filelink:`pkg/ctrl` and will be treated in the next section.
As an example we use the cost function of the verification experiment
:filelink:`tutorial_tracer_adjsens <verification/tutorial_tracer_adjsens>`
(:numref:`sec_tracer_adj_sens`), which is computed from a passive tracer.

::

          the_model_main
          |
          |-- initialise_fixed
          |   |
          |   |-- packages_readparms
          |   |   |
          |   |   |-- cost_readparms
          |   |   o
          |   |
          |   |-- packages_init_fixed
          |       |
          |       |-- cost_init_fixed
          |       o
          |
          |-- the_main_loop
         ...  |
              |-- initialise_varia
              |   |
              |   |-- cost_init_varia
              |   o
              |
              |-- do iloop = 1,nTimeSteps
              |      |-- forward_step
              |      |   |
              |      |   |-- cost_tile
              |      |       |
              |      |       |-- cost_tracer
              |   end do
              |
              |-- cost_final
              o

Enabling the package
~~~~~~~~~~~~~~~~~~~~

:filelink:`pkg/cost <pkg/cost>` is enabled by adding the line ``cost`` to your
file ``packages.conf`` (see :numref:`using_packages`).

In general the following packages ought to be enabled simultaneously:
:filelink:`pkg/autodiff <pkg/autodiff>`, :filelink:`pkg/ctrl <pkg/ctrl>`, and
:filelink:`pkg/cost`. The basic CPP option to enable the cost function is
:varlink:`ALLOW_COST`. Each specific cost function contribution has its own
option. For the present example the option is :varlink:`ALLOW_COST_TRACER`. All
cost-specific options are set in :filelink:`COST_OPTIONS.h
<pkg/cost/COST_OPTIONS.h>`.

Initialization
~~~~~~~~~~~~~~

The initialization of :filelink:`pkg/cost` is readily enabled as soon as
the CPP option :varlink:`ALLOW_COST` is defined.

-  The S/R :filelink:`cost_readparms.F </pkg/cost/cost_readparms.F>`
   reads runtime flags and parameters from file ``data.cost``.
   For the present example the only relevant parameter read is
   :varlink:`mult_tracer`. This multiplier enables different cost function
   contributions to be switched on (``= 1.``) or off (``= 0.``) at runtime.
   For more complex cost functions which involve model vs. data
   misfits, the corresponding data filenames and data specifications
   (start date and time, period, ...) are read in this S/R.

-  The S/R :filelink:`cost_init_varia.F </pkg/cost/cost_init_varia.F>`
   initializes the different cost function contributions. The
   contribution for the present example is :varlink:`objf_tracer` which is
   defined on each tile (bi,bj).

Accumulation
~~~~~~~~~~~~

The ’driver’ routine :filelink:`cost_tile.F </pkg/cost/cost_tile.F>`
is called at the end of each time
step. Within this ’driver’ routine, S/R are called for each of the
chosen cost function contributions. In the present example
(:varlink:`ALLOW_COST_TRACER`), S/R :filelink:`cost_tracer.F </pkg/cost/cost_tracer.F>` is called. It accumulates
:varlink:`objf_tracer` according to :eq:`cost_tracer`.

.. _sec_ad_finalize_contribtuions:

Finalize all contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~

At the end of the forward integration S/R :filelink:`cost_final.F </pkg/cost/cost_final.F>` is called. It
accumulates the total cost function :varlink:`fc` from each contribution and
sums over all tiles:

.. math::
   {\cal J} \, = \,
   {\rm fc} \, = \,
   {\rm mult\_tracer} \sum_{\text{global sum}} \sum_{bi,\,bj}^{nSx,\,nSy}
   {\rm objf\_tracer}(bi,bj) \, + \, ...

The total cost function :varlink:`fc` is the ’dependent’ variable that is
named in the argument list of the AD tool (:numref:`adoptfile`).

.. _the_ctrl_vars:

The control variables (independent variables)
---------------------------------------------

The control variables are a subset of the model input (initial
conditions, boundary conditions, model parameters). Here we identify
them with the variable :math:`\vec{u}`. All intermediate variables
whose derivative with respect to control variables do not vanish are called
active variables. All subroutines whose derivative with respect to the control
variables don’t vanish are called active routines. Read and write
operations from and to file can be viewed as variable assignments.
Therefore, files to which active variables are written and from which
active variables are read are called active files. All aspects relevant
to the treatment of the control variables (parameter setting,
initialization, perturbation) are controlled by the package :filelink:`pkg/ctrl`.
The control variables that are available and their run-time parameters are
described in :numref:`sec:pkg:ctrl`; this section describes how a control
variable enters the differentiated code.

::

          the_model_main
          |
          |-- initialise_fixed
          |   |
          |   |-- packages_readparms
          |   |   |
          |   |   |-- ctrl_readparms
          |   |   o
          |   |
          |   |-- packages_init_fixed
          |       |
          |       |-- ctrl_init_fixed
          |       o
          |
          |-- ctrl_unpack
          |
          |-- the_main_loop
          |   |
          |   |-- initialise_varia
          |   |   |
          |   |   |-- packages_init_variables
          |   |       |
          |   |       |-- ctrl_init_variables
          |   |           |
          |   |           |-- ctrl_map_ini_genarr
          |   |           |-- ctrl_map_ini_gentim2d
          |   |           o
          |   |
          |   |-- do iloop = 1,nTimeSteps
          |   |      |-- forward_step
          |   |      |   |
          |   |      |   |-- load_fields_driver
          |   |      |   |   |
          |   |      |   |   |-- ctrl_map_gentim2d
          |   |      |   |
          |   |      |   |-- ctrl_map_forcing
          |   |   end do
          |   o
          |
          |-- ctrl_pack
          o

Enabling the package
~~~~~~~~~~~~~~~~~~~~

Package :filelink:`pkg/ctrl` is enabled by adding the line ``ctrl`` to your
file ``packages.conf``. The generic control variables are enabled by the CPP
options :varlink:`ALLOW_GENARR2D_CONTROL` and :varlink:`ALLOW_GENARR3D_CONTROL`
(2-D and 3-D fields that are constant in time, such as initial conditions
and model parameters) and :varlink:`ALLOW_GENTIM2D_CONTROL` (2-D fields that
vary in time, such as the surface forcing) in :filelink:`CTRL_OPTIONS.h
<pkg/ctrl/CTRL_OPTIONS.h>`. The individual control variables are then
chosen at run time in ``data.ctrl`` (see :numref:`gen_ctrl`).

Initialization
~~~~~~~~~~~~~~

- The S/R :filelink:`ctrl_readparms.F </pkg/ctrl/ctrl_readparms.F>`
  reads runtime flags and parameters from file ``data.ctrl``.
  The file contains the file names of each
  control variable that is used. In S/R :filelink:`ctrl_init_wet.F
  </pkg/ctrl/ctrl_init_wet.F>`, which is called from S/R
  :filelink:`ctrl_init_fixed.F </pkg/ctrl/ctrl_init_fixed.F>`, the number of wet
  points for each control variable and the net dimension of the space
  of control variables (counting wet points only) :varlink:`nvarlength` is
  determined. The numbers of wet points for each tile (bi,bj) and
  vertical layer k are determined for the three relevant
  categories on the C-grid: :varlink:`nWetCtile` for tracer fields,
  :varlink:`nWetWtile` for zonal velocity fields, :varlink:`nWetStile` for
  meridional velocity fields.

- Two important issues related to the handling of the control
  variables in MITgcm need to be addressed. First, in order to save
  memory, the control variable arrays are not kept in memory, but
  rather read from file and added to the initial fields during the
  model initialization phase. Similarly, the corresponding adjoint
  fields which represent the gradient of the cost function with respect to the
  control variables are written to file at the end of the adjoint
  integration. Second, in addition to the files holding the 2-D
  and 3-D control variables and the corresponding cost gradients,
  a 1-D control vector and gradient vector are written to file.
  They contain only the wet points of the control variables and the
  corresponding gradient. This leads to a significant data
  compression. Furthermore, an option is available
  (:varlink:`ALLOW_NONDIMENSIONAL_CONTROL_IO`) to non-dimensionalize the
  control and gradient vector, which otherwise would contain
  different pieces of different magnitudes and units. Finally, the
  control and gradient vector can be passed to a minimization routine
  if an update of the control variables is sought as part of a
  minimization exercise (:numref:`sec_ad_optim`).

The files holding the fields of the control variables and of the gradient
are generated and initialized in S/R :filelink:`ctrl_init_ctrlvar.F
</pkg/ctrl/ctrl_init_ctrlvar.F>`, which is called from S/R
:filelink:`ctrl_init_fixed.F </pkg/ctrl/ctrl_init_fixed.F>`. The control
and gradient vectors are read and written in S/R :filelink:`ctrl_unpack.F
</pkg/ctrl/ctrl_unpack.F>` and S/R :filelink:`ctrl_pack.F
</pkg/ctrl/ctrl_pack.F>`.

.. _sec_ad_perturbation:

Perturbation of the independent variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dependency flow for differentiation with respect to the controls starts with
adding a perturbation onto the input variable, thus defining the
independent or control variables for the AD tool. Three types of controls may be
considered:

- Initial conditions. Consider as an example the initial temperature
  :varlink:`theta` as control variable. After :varlink:`theta` has been
  initialized in :filelink:`ini_fields.F <model/src/ini_fields.F>`, a
  perturbation anomaly is added to the field in S/R ``CTRL_MAP_GENARR3D``
  (file :filelink:`ctrl_map_genarr.F </pkg/ctrl/ctrl_map_genarr.F>`), which
  is called from S/R :filelink:`ctrl_map_ini_genarr.F
  </pkg/ctrl/ctrl_map_ini_genarr.F>`:

  .. math::
        \begin{aligned}
        u         & = \, u_{[0]} \, + \, \Delta u \\
        {\bf theta}(...) & = \, {\bf theta_{ini}}(...) \, + \, {\bf xx\_theta}(...)
        \end{aligned}
        :label: perturb

  :varlink:`xx_theta` is a 3-D global field holding the perturbation. In
  the case of a simple sensitivity study this field is identical to
  zero. However, it’s specification is essential in the context of
  automatic differentiation since the AD tool treats the corresponding line
  in the code symbolically when determining the differentiation chain
  and its origin.

  Now, as mentioned above, MITgcm avoids maintaining an array for each
  control variable by reading the perturbation to a temporary array
  from file. To ensure the symbolic link to be recognized by the AD tool,
  an ’active read’ routine of the adjoint support package
  :filelink:`pkg/autodiff` is invoked, and the read-procedure is tagged
  with a dummy variable, here an element of ``xx_genarr3d_dummy``. The
  operation (to perturb) in the code takes on the form

  ::

              CALL ACTIVE_READ_XYZ( fnamegenIn, xx_gen, 1, doglobalread,
            &      ladinit, optimcycle, myThid, xx_genarr3d_dummy(iarr) )

              fld(...) = fld(...) + xx_gen(...)

  where ``fld`` is the model field that is passed to the routine, here
  :varlink:`theta`. The dummy variables, one array for each kind of generic
  control variable (``xx_genarr2d_dummy``, ``xx_genarr3d_dummy``,
  ``xx_gentim2d_dummy``), are the independent variables that are named in
  the argument list of the AD tool (:numref:`adoptfile`).

  Note that reading an active variable corresponds to a variable
  assignment. Its derivative corresponds to a write statement of the
  adjoint variable, followed by a reset. The ’active file’ routines
  have been designed to support active read and corresponding adjoint
  active write operations (and vice versa). They are not differentiated
  by the AD tool; their tangent linear and adjoint versions are written by
  hand (:numref:`sec_ad_hidden`).

- Boundary conditions. The handling of boundary values as control
  variables proceeds exactly analogous to the initial values, with the
  perturbation read in S/R :filelink:`ctrl_map_gentim2d.F
  </pkg/ctrl/ctrl_map_gentim2d.F>` and added to the forcing fields in S/R
  :filelink:`ctrl_map_forcing.F </pkg/ctrl/ctrl_map_forcing.F>` or in
  :filelink:`pkg/exf`. Note however
  an important difference: Since the boundary values are time
  dependent with a new forcing field applied at each time step, the
  general problem may be thought of as a new control variable at each
  time step (or, if the perturbation is averaged over a certain
  period, at each :math:`N` timesteps), i.e.,

  .. math::
        u_{\rm forcing} \, = \,
        \{ \, u_{\rm forcing} ( t_n ) \, \}_{
        n \, = \, 1, \ldots , {\rm nTimeSteps} }

  In MITgcm the perturbation is defined at regular intervals (a period
  that is set in ``data.ctrl``) and interpolated in time between them.

- Model parameters. The handling proceeds along the same lines as the
  initial value sensitivity. The mixing parameters, for example, are
  available as the 3-D control variables ``xx_diffkr`` and ``xx_kapgm``,
  which are added in
  :filelink:`ctrl_map_ini_genarr.F </pkg/ctrl/ctrl_map_ini_genarr.F>`.

.. _sec_ad_code:

Differentiated and hand-written code
------------------------------------

Not all of the code below the top-level routine is differentiated by the AD
tool. This section describes how the routines that the tool differentiates
are selected, and what is done about the others.

The list ``AD_FILES`` and ``*_ad_diff.list`` files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not all routines are presented to the AD tool. Routines typically hidden
are diagnostics routines which do not influence the cost function, but
may create artificial flow dependencies such as I/O of active variables.

:filelink:`genmake2 <tools/genmake2>` generates a list (or variable) ``AD_FILES``
that contains all routines that are shown to the AD tool.
This list is put together from all files with suffix ``_ad_diff.list``
that :filelink:`genmake2 <tools/genmake2>` finds in its search directories.
The list file for the core MITgcm routines is :filelink:`model/src/model_ad_diff.list`.
Of the MITgcm WRAPPER (:numref:`wrapper`), only the routines that are listed
in :filelink:`eesupp/src/eesupp_ad_diff.list` and
:filelink:`eesupp/src/exch_ad_diff.list` are shown to the AD tool. The others are either not
visible at all to the AD code, or hand-written AD code is available (see
below).

A package with routines that are to be differentiated contains its package-specific list file
``«PKG»_ad_diff.list``. For example, :filelink:`pkg/ptracers` contains the file
:filelink:`ptracers_ad_diff.list <pkg/ptracers/ptracers_ad_diff.list>`.
Thus, enabling a package will automatically
extend the ``AD_FILES`` list of :filelink:`genmake2 <tools/genmake2>` to incorporate the
package-specific routines. Note that you will need to regenerate the
makefile if you enable a package (e.g., by adding it to ``packages.conf``)
and a ``Makefile`` already exists.

.. _sec_ad_hidden:

Routines hidden from the AD tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A routine that is not contained in the ``AD_FILES`` list but is called in a
part of the code that the AD tool does see is ’hidden’ from the tool. For
such a routine the tool needs to be told

-  which subroutine arguments are input/output

-  which subroutine arguments are active

and, if the routine takes part in the computation of the cost function,
its derivative code has to be supplied by hand. Examples are the active
file routines (:numref:`sec_ad_perturbation`) and the global sum routines
of the MITgcm WRAPPER (:numref:`wrapper`). Each tool has its own form for both
(:numref:`tab_ad_hidden`): the dependencies are described by flow
directives for TAF and by the entries of an external library file for
Tapenade, and a hand-written derivative routine carries the name and the
argument list that the tool generates the call for
(:numref:`tab_ad_naming`).

.. table:: Where the hidden routines are described and where their
           hand-written derivative code resides.
   :name: tab_ad_hidden

   +---------------------------+-------------------------------------------+-------------------------------------------+
   |                           | TAF                                       | Tapenade                                  |
   +===========================+===========================================+===========================================+
   | description of the        | flow directives in ``.flow`` files        | external library file                     |
   | dependencies              | (:numref:`ad_taf_flow`)                   | :filelink:`tools/TAP_support/flow_tap`    |
   |                           |                                           | (:numref:`ad_tap_handwritten`)            |
   +---------------------------+-------------------------------------------+-------------------------------------------+
   | hand-written derivative   | :filelink:`pkg/autodiff` (files ``ad*.F``,| :filelink:`pkg/tapenade`,                 |
   | code                      | ``*_ad.F``, ``g_*.F``, ``*_g.F``),        | :filelink:`eesupp/src` (``exch_tap_b.F``, |
   |                           | :filelink:`eesupp/src` (``*_ad.F``) and   | ``exch_tap_d.F``) and                     |
   |                           | the package directories                   | the package directories                   |
   +---------------------------+-------------------------------------------+-------------------------------------------+

.. _ad_output_hooks:

Hooks
~~~~~

A special case of hidden routines are hooks: subroutines that do nothing in
the forward model and whose adjoint and tangent linear counterparts are
written by hand. They are the means to execute code in the reverse sweep
that has no counterpart in the forward model. The forward model calls four
hooks that are used with both AD tools: S/R ``AUTODIFF_INADMODE_UNSET``
near the start of each time step, S/R ``DUMMY_IN_STEPPING`` before the
dynamics and S/R ``AUTODIFF_INADMODE_SET`` at the end of the step, in
:filelink:`forward_step.F <model/src/forward_step.F>`, and S/R
``DUMMY_FOR_ETAN`` in :filelink:`integr_continuity.F
<model/src/integr_continuity.F>` after the free-surface update, separately
because the adjoint of the free surface is one time step out of phase with
the rest of the state. Since the reverse sweep passes through the code of a
time step in reverse order, the adjoint of a hook that is called at the
beginning of the time step is executed at the end of the adjoint time step,
and vice versa.

The adjoints of ``DUMMY_IN_STEPPING`` and ``DUMMY_FOR_ETAN``, which are
called only if :varlink:`ALLOW_AUTODIFF_MONITOR` is defined, write the
adjoint state (:numref:`ad_output_dumps_diags`); those of the other two
apply and revert the adjoint-mode parameter switches
(:numref:`ad_inadmode_switches`). A hook has no active argument, and each AD
tool is made to keep its call in the generated code in its own way: see
:numref:`ad_taf_hooks` for TAF and :numref:`ad_tap_handwritten` for
Tapenade.

.. _sec_ad_build:

The AD build process
--------------------

The build process of an AD code is very similar to building the forward
model (:numref:`building_code`). :filelink:`genmake2 <tools/genmake2>`
generates a ``Makefile`` with additional ``make`` targets for the adjoint
and for the tangent linear model; they differ between the two AD tools
(:numref:`tab_ad_build`).

.. table:: Building the adjoint and the tangent linear model with TAF and
           with Tapenade.
   :name: tab_ad_build

   +-----------------------------+-----------------------------------------+-----------------------------------------+
   |                             | TAF                                     | Tapenade                                |
   +=============================+=========================================+=========================================+
   | :filelink:`genmake2         | none (default)                          | ``-tap``                                |
   | <tools/genmake2>` option    |                                         |                                         |
   +-----------------------------+-----------------------------------------+-----------------------------------------+
   | default AD options file     | :filelink:`adjoint_default              | :filelink:`adjoint_tap                  |
   |                             | <tools/adjoint_options/adjoint_default>`| <tools/adjoint_options/adjoint_tap>`    |
   +-----------------------------+-----------------------------------------+-----------------------------------------+
   | adjoint model               | ``make adall``                          | ``make tap_adj``                        |
   +-----------------------------+-----------------------------------------+-----------------------------------------+
   | tangent linear model        | ``make ftlall``                         | ``make tap_tlm``                        |
   +-----------------------------+-----------------------------------------+-----------------------------------------+
   | described in                | :numref:`building_adcode_using_taf`     | :numref:`ad_tap_build`                  |
   +-----------------------------+-----------------------------------------+-----------------------------------------+

With either tool, such a target consists of the following procedures:

#. A header file ``AD_CONFIG.h`` is generated which contains a CPP option
   on which code ought to be generated. Depending on the ``make`` target,
   the contents is one of the following:

   -  #define :varlink:`ALLOW_ADJOINT_RUN`

   -  #define :varlink:`ALLOW_TANGENTLINEAR_RUN`

#. The source files of the list ``AD_FILES`` (:numref:`sec_ad_code`) are
   preprocessed and passed to the AD tool together with the description of
   the hidden routines (:numref:`sec_ad_hidden`) and the AD tool flags
   (:numref:`adoptfile`).

#. The AD tool writes the resulting tangent linear or adjoint code to new
   source files.

#. All routines are compiled and an executable is generated
   (:numref:`tab_ad_naming`).

.. _adoptfile:

Changing the default AD tool flags: AD options files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The flags with which the AD tool is invoked are set in an AD options file,
which is similar to the platform-specific build options file (see
:numref:`genmake2_optfiles`). The default files reside in
:filelink:`tools/adjoint_options` (:numref:`tab_ad_build`); a different
file is selected with the :filelink:`genmake2 <tools/genmake2>` option
``-adoptfile`` (or ``-adof``). Among these flags are the top-level routine,
the independent variables and the dependent variable
(:numref:`sec_ad_perturbation`, :numref:`sec_ad_finalize_contribtuions`).
For TAF they are given as

::

    -toplevel 'the_main_loop'
    -input '... xx_genarr2d_dummy xx_genarr3d_dummy xx_gentim2d_dummy'
    -output 'fc'

where the dots stand for the dummy variables of the open boundary controls,
which are named for TAF only, and for Tapenade as

::

    -head 'the_main_loop(fc)/(xx_genarr3d_dummy, xx_genarr2d_dummy, xx_gentim2d_dummy)'

Additional flags can be passed to the AD tool without a new AD options file
through the :filelink:`genmake2 <tools/genmake2>` options ``-taf_extra`` and
``-tap_extra``.

.. _sec_ad_storing:

Storing the forward trajectory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reverse sweep requires intermediate results of the forward sweep
(:numref:`sec_autodiff_storage_v_recompute`). The two AD tools organize
their storage differently. TAF stores the variables that are named in store
directives, which are part of the MITgcm source code, and combines them
with a multi-level checkpointing of the time stepping loop that is
configured at compile time (:numref:`ad_taf_checkpointing`). Tapenade
requires no store directives: it determines by itself which values have to be
saved, keeps them on a stack, and by default checkpoints every subroutine
call (:numref:`ad_tap_build_detail`).

.. _sec_autodiff_output_adj_vars:

.. _ad_output:

Output of adjoint variables and gradient
----------------------------------------

An adjoint run writes the gradient of the cost function with respect to
the control variables. In addition, the adjoint variables of the model
state can be written at intermediate times; a tangent linear run writes
the corresponding tangent linear variables.

The gradient
~~~~~~~~~~~~

-  In :filelink:`ctrl_map_ini_genarr.F </pkg/ctrl/ctrl_map_ini_genarr.F>`,
   :filelink:`ctrl_map_gentim2d.F </pkg/ctrl/ctrl_map_gentim2d.F>`:

   -  The control variable fields ``xx_«...»``: before the forward integration, the control variables are read
      from file ``xx_«...»`` and added to the model field.

   -  The adjoint variable fields ``adxx_«...»``, i.e., the gradient
      :math:`\nabla _{u}{\cal J}` for each control variable:
      after the adjoint integration the corresponding adjoint
      variables are written to ``adxx_«...»``.

-  In :filelink:`ctrl_unpack.F </pkg/ctrl/ctrl_unpack.F>`, :filelink:`ctrl_pack.F </pkg/ctrl/ctrl_pack.F>`:

   -  The control vector ``vector_ctrl``:
      before the forward integration, the updated
      compressed control vector, if an optimization has provided one, is read and
      distributed to 2-D and 3-D control variable fields.

   -  The gradient vector ``vector_grad``:
      at the very end of the adjoint integration, the 2-D and
      3-D adjoint variables are read, compressed to a single vector
      and written to file.

   The names of the two vector files are composed from the run-time
   parameters :varlink:`ctrlname` and :varlink:`costname`; by default they
   are ``ecco_ctrl_MIT_CE_000.opt«kkkk»`` and
   ``ecco_cost_MIT_CE_000.opt«kkkk»``, where «kkkk» is the number of the
   optimization iteration.

.. _ad_output_dumps_diags:

Dumps and diagnostics of the adjoint state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to writing the gradient at the end of the forward/adjoint
integration, many more adjoint variables of the model state at
intermediate times can be written. With :varlink:`ALLOW_AUTODIFF_MONITOR`
defined in :filelink:`AUTODIFF_OPTIONS.h <pkg/autodiff/AUTODIFF_OPTIONS.h>`,
the adjoint state is available in two forms:

- as binary files ``ADJtheta.*``, ``ADJsalt.*``, ``ADJuvel.*``, ..., written
  every :varlink:`adjDumpFreq` seconds (namelist ``PARM03`` of ``data``) by the
  hand-written adjoint of S/R ``DUMMY_IN_STEPPING``, and ``ADJetan.*`` by that
  of S/R ``DUMMY_FOR_ETAN`` (:numref:`ad_output_hooks`); unless
  :varlink:`dumpAdVarExch` ``= 0`` (``data.autodiff``, the default is 2), the
  adjoint contributions in the tile overlaps are added to the interior points
  that own them before a field is written;
- as the diagnostics ``ADJtheta``, ``ADJuvel``, ... of
  :filelink:`pkg/diagnostics`, filled by the same routines whenever
  ``data.diagnostics`` requests them (see
  :ref:`outp_pkg_diagnostics_adjoint_vars` for the names and the setup).

A tangent-linear model writes the ``G_J*`` counterparts of the binary files.

With TAF, the fields written are those of
:filelink:`addummy_in_stepping.F <pkg/autodiff/addummy_in_stepping.F>` and of
the package routines it calls
(:filelink:`seaice_ad_dump.F <pkg/seaice/seaice_ad_dump.F>`,
:filelink:`ptracers_ad_dump.F <pkg/ptracers/ptracers_ad_dump.F>`), and the
:filelink:`pkg/exf` snapshots of
:filelink:`exf_adjoint_snapshots_ad.F <pkg/exf/exf_adjoint_snapshots_ad.F>`.
With Tapenade, they are ``ADJetan`` and those of
:filelink:`dummy_in_stepping_tap.F <pkg/tapenade/dummy_in_stepping_tap.F>`:
``ADJtheta``, ``ADJsalt``, ``ADJuvel``, ``ADJvvel`` and ``ADJwvel``; if
neither :filelink:`pkg/exf` nor :filelink:`pkg/seaice` is used also
``ADJtaux``, ``ADJtauy``, ``ADJqnet``, ``ADJempr`` and, with
``SHORTWAVE_HEATING``, ``ADJqsw``; and ``ADJdiffkr`` with
``ALLOW_DIFFKR_CONTROL``. These are the names of the binary files; the
diagnostics that correspond to ``ADJempr`` and ``ADJdiffkr`` are named
``ADJempmr`` and ``ADJdifkr``. The adjoint fields of :filelink:`pkg/seaice`,
:filelink:`pkg/ptracers`, :filelink:`pkg/exf`, :filelink:`pkg/gmredi`,
:filelink:`pkg/ggl90` and :filelink:`pkg/shelfice`, and the bottom-drag and
eddy-streamfunction sensitivities, are not written with Tapenade, and their
diagnostics are not available in a build with :filelink:`pkg/tapenade`.

How a field is added to this output depends on the AD tool
(:numref:`ad_taf_hooks`, :numref:`ad_tap_handwritten`). In both cases,
register the diagnostic in the package's ``*_diagnostics_init.F`` with ``A``
in the fourth position of ``diagCode``
(:ref:`outp_pkg_diagnostics_adjoint_vars`).

.. _ad_inadmode_switches:

Adjoint-mode parameter switches
-------------------------------

The adjoint-mode parameters of ``data.autodiff`` (namelist
``AUTODIFF_PARM01``;
:filelink:`autodiff_readparms.F <pkg/autodiff/autodiff_readparms.F>` has the
complete list with defaults) change the model physics in the backward sweep
only, so that the adjoint of a simplified or more dissipative model is
computed along the trajectory of the full forward model. S/R
``ADAUTODIFF_INADMODE_SET``
(:filelink:`autodiff_inadmode_set_ad.F <pkg/autodiff/autodiff_inadmode_set_ad.F>`)
applies them at the start of every backward time step and S/R
``ADAUTODIFF_INADMODE_UNSET``
(:filelink:`autodiff_inadmode_unset_ad.F <pkg/autodiff/autodiff_inadmode_unset_ad.F>`)
reverts them at its end; the logical :varlink:`inAdMode` is ``.TRUE.`` in
between.

- :varlink:`useKPPinAdMode`, :varlink:`useGMRediInAdMode`,
  :varlink:`useSEAICEinAdMode`, :varlink:`useGGL90inAdMode` and
  :varlink:`useSALT_PLUMEinAdMode` replace the corresponding ``use«PKG»``
  flag in the backward sweep, so that a package can be left out of the
  adjoint while it remains in the forward sweep.
- :varlink:`useApproxAdvectionInAdMode` (also set by :varlink:`inAdExact`
  ``= .FALSE.``) replaces the flux-limited third-order direct space-time
  advection scheme (33) by its unlimited form (30) in the backward sweep;
  implicit vertical advection is not affected.
- :varlink:`viscFacInAd` scales the lateral viscosities read from file in
  the backward sweep, and its counterpart :varlink:`viscFacInFw` in the
  forward sweep (both require the CPP option
  :varlink:`AUTODIFF_ALLOW_VISCFACADJ`).
- :filelink:`pkg/seaice` has further switches of its own.

The switches work with either AD tool: the two routines are called in the
reverse sweep through the hooks S/R ``AUTODIFF_INADMODE_SET`` and S/R
``AUTODIFF_INADMODE_UNSET`` (:numref:`ad_output_hooks`).

Whether a switched value reaches a given routine depends on how that
routine is differentiated. In Tapenade's default mode a routine is
re-executed inside its adjoint, after the switch has acted, so
:varlink:`viscFacInAd` reaches the viscosities recomputed there. A routine
differentiated in split mode (``-nocheckpoint``) instead replays the values
recorded in the forward sweep, before any switch, and the factor does not
reach it. This is the situation of a TAF adjoint in which the viscosities
are stored rather than recomputed, which is why the store directives for
them in :filelink:`mom_vecinv.F <pkg/mom_vecinv/mom_vecinv.F>` and
:filelink:`mom_fluxform.F <pkg/mom_fluxform/mom_fluxform.F>` are left out
when ``AUTODIFF_ALLOW_VISCFACADJ`` is defined.

.. _sec_ad_optim:

Control variable handling for optimization applications
-------------------------------------------------------

In optimization mode the cost function :math:`{\cal J}(u)` is sought
to be minimized with respect to a set of control variables
:math:`\delta {\cal J} \, = \, 0`, in an iterative manner. The
gradient :math:`\nabla _{u}{\cal J} |_{u_{[k]}}` together with the
value of the cost function itself :math:`{\cal J}(u_{[k]})` at
iteration step :math:`k` serve as input to a minimization routine
(e.g. quasi-Newton method, conjugate gradient, ... (Gilbert and Lemaréchal, 1989
:cite:`gil-lem:89`) to compute an update in the control
variable for iteration step :math:`k+1`:

.. math::
   u_{[k+1]} \, = \,  u_{[0]} \, + \, \Delta u_{[k+1]}
   \quad \mbox{satisfying} \quad
    {\cal J} \left( u_{[k+1]} \right) \, < \, {\cal J} \left( u_{[k]} \right)

:math:`u_{[k+1]}` then serves as input for a forward/adjoint run to
determine :math:`{\cal J}` and :math:`\nabla _{u}{\cal J}` at
iteration step :math:`k+1`. :numref:`forward-adj_flow` sketches the flow
between forward/adjoint model and the minimization routine.

 .. figure:: figs/forward-adj_flow.*
    :width: 100%
    :align: center
    :alt: flow between forward/adjoint model and the minimization
    :name: forward-adj_flow

    Flow between the forward/adjoint model and the minimization routine.

The routines :filelink:`ctrl_unpack.F </pkg/ctrl/ctrl_unpack.F>` and
:filelink:`ctrl_pack.F </pkg/ctrl/ctrl_pack.F>` provide the link between
the model and the minimization routine. As described in
:numref:`the_ctrl_vars` the :filelink:`ctrl_unpack.F </pkg/ctrl/ctrl_unpack.F>`
and :filelink:`ctrl_pack.F </pkg/ctrl/ctrl_pack.F>` routines read and write
control and gradient vectors which are compressed to contain only wet
points, in addition to the full 2-D and 3-D fields. The
corresponding I/O flow is shown in :numref:`forward-adj_io`:

 .. figure:: figs/forward-adj_io.*
    :width: 100%
    :align: center
    :alt: forward/adjoint model I/O
    :name: forward-adj_io

    Flow chart showing I/O in the forward/adjoint model.

:filelink:`ctrl_unpack.F </pkg/ctrl/ctrl_unpack.F>` reads the updated control
vector ``vector_ctrl_<k>``. It distributes the different control variables to
2-D and 3-D files ``xx_«...»<k>``. At the start of the forward integration the
control variables are read from ``xx_«...»<k>`` and added to the field.
Correspondingly, at the end of the adjoint integration the adjoint fields are
written to ``adxx_«...»<k>``, again via the active file routines. Finally,
:filelink:`ctrl_pack.F </pkg/ctrl/ctrl_pack.F>` collects all adjoint files and
writes them to the compressed vector file ``vector_grad_<k>``.

.. _ad_gradient_check:

The gradient check package
==========================

An indispensable test to validate the gradient computed via the adjoint
is a comparison against finite difference gradients. The gradient check
package :filelink:`pkg/grdchk` enables such tests in a straightforward and easy
manner. The driver routine :filelink:`grdchk_main.F <pkg/grdchk/grdchk_main.F>` is called from
:filelink:`the_model_main.F <model/src/the_model_main.F>` after
the gradient has been computed via the adjoint
model (cf. the flow chart in :numref:`sec_ad_run_structure`).

The gradient check proceeds as follows: The :math:`i-`\ th component of
the gradient :math:`(\nabla _{u}{\cal J}^T)_i` is compared with the
following finite-difference gradient:

.. math::
   \left(\nabla _{u}{\cal J}^T  \right)_i \quad \text{ vs. } \quad
   \frac{\partial {\cal J}}{\partial u_i} \, = \,
   \frac{ {\cal J}(u_i + \epsilon) - {\cal J}(u_i)}{\epsilon}

With :varlink:`useCentralDiff` ``= .TRUE.`` (the default) the centered
difference :math:`\left( {\cal J}(u_i + \epsilon) - {\cal J}(u_i - \epsilon)
\right) / 2 \epsilon` is used instead.

A gradient check at point :math:`u_i` may generally considered to be
successful if the deviation of the ratio between the finite difference and
the adjoint gradient from unity is less than 1 percent,

.. math::
   1 \, - \,
   \frac{({\rm grad}{\cal J})_i (\text{finite difference})}
   {({\rm grad}{\cal J})_i (\text{adjoint})} \, < 1 \%

Code configuration
------------------

The package is compiled by adding ``grdchk`` to ``packages.conf``; it is
part of the package group ``adjoint`` (:numref:`sec_ad_general_setup`).

The relevant runtime flags are set in the files:

- ``data.pkg``
  - Set :varlink:`useGrdchk` ``= .TRUE.``

-  ``data.grdchk``

   -  :varlink:`grdchk_eps`

   -  :varlink:`nbeg`

   -  :varlink:`nstep`

   -  :varlink:`nend`

   -  :varlink:`grdchkvarindex`

::

       the_model_main
       |
       |-- ctrl_unpack
       |-- adthe_main_loop            - unperturbed cost function and
       |-- ctrl_pack                    adjoint gradient are computed here
       |
       |-- grdchk_main
           |
           |-- grdchk_get_mask
           |-- do icomp=...           - loop over control vector elements
           |   |
           |   |-- grdchk_loc         - determine location of icomp on grid
           |   |
           |   |-- grdchk_getadxx     - get gradient component calculated
           |   |                        via adjoint
           |   |-- grdchk_getxx       - get control vector component from file
           |   |                        perturb it and write back to file
           |   |-- the_main_loop      - forward run and cost evaluation
           |   |                        with perturbed control vector element
           |   |-- grdchk_setxx       - Reset control vector element
           |   |
           |   |-- calculate ratio of adj. vs. finite difference gradient
           |
           |-- grdchk_print           - print results

The flow chart shows an adjoint run. In a tangent linear run, the gradient
check package compares the finite-difference gradient with the tangent
linear gradient in the same manner: for each of the selected components it
calls the tangent linear model, S/R ``g_the_main_loop`` with TAF and S/R
``the_main_loop_d`` with Tapenade, with a unit perturbation of that
component. With Tapenade the tangent linear model is called from the
gradient check package only, so that a tangent linear run requires
:varlink:`useGrdchk` ``= .TRUE.``.

.. _ad_taf:

Adjoint code generation using TAF
=================================

TAF and its predecessor TAMC are the AD tools for which the tangent linear
and adjoint capabilities of MITgcm were first developed. This section
describes what is specific to TAF: the build procedure, the directives
through which TAF is told about hidden routines and about the storing of
the forward trajectory, the hand-written adjoint code, and the divided
adjoint. What TAF has in common with other AD tools is described in
:numref:`sec_ad_tlm_and_adm`.

.. _building_adcode_using_taf:

Building the AD code using TAF
------------------------------

The build process of an AD code is very similar to building the forward
model (:numref:`sec_ad_build`). Depending on which AD code one wishes to
generate, the following make targets are available:

+------------------+------------------------+----------------------------------------------------------------------------------+
| *AD-target*      | *output*               | *description*                                                                    |
+==================+========================+==================================================================================+
| ``adtaf``        | ``ad_taf_output.f``    | generates code for the adjoint model (ADM) using TAF,                            |
|                  |                        | without compilation                                                              |
+------------------+------------------------+----------------------------------------------------------------------------------+
| ``adall``        | ``mitgcmuv_ad``        | generates code for the ADM using TAF                                             |
|                  |                        | and compiles all code                                                            |
+------------------+------------------------+----------------------------------------------------------------------------------+
| ``ftltaf``       | ``ftl_taf_output.f``   | generates code for the tangent linear model (TLM) using TAF,                     |
|                  |                        | without compilation                                                              |
+------------------+------------------------+----------------------------------------------------------------------------------+
| ``ftlall``       | ``mitgcmuv_ftl``       | generates code for the TLM using TAF                                             |
|                  |                        | and compiles all code                                                            |
+------------------+------------------------+----------------------------------------------------------------------------------+
| ``svdtaf``       | ``ad_taf_output.f``,   | generates code for both ADM and TLM using TAF, for                               |
|                  | ``ftl_taf_output.f``   | singular value decomposition (SVD) type calculations                             |
+------------------+------------------------+----------------------------------------------------------------------------------+
| ``svdall``       | ``mitgcmuv_svd``       | compiles the code generated by ``make svdtaf``                                   |
|                  |                        | and all other code                                                               |
+------------------+------------------------+----------------------------------------------------------------------------------+

For example, to generate the adjoint model using TAF after routines (``.F``)
or headers (``.h``) have been modified, but without compilation,
type ``make adtaf``.

A typical full build process to generate the ADM via TAF would look like
follows:

::

    % mkdir build
    % cd build
    % ../../../tools/genmake2 -mods=../code_ad [ -nocat4ad ]
    % make depend
    % make adall

The AD build process in detail
------------------------------

For TAF, the ``make adall`` target consists of the following procedures
(cf. :numref:`sec_ad_build`):

#. A header file ``AD_CONFIG.h`` is generated which contains the CPP option
   #define :varlink:`ALLOW_ADJOINT_RUN`.

#. If ``-nocat4ad`` is not specified, a single file ``ad_input_code.f`` is
   concatenated consisting of all ``.flow`` files that are part of the list
   ``AD_FLOW_FILES`` and all ``.f`` files that are part of the list
   ``AD_FILES``.

#. TAF is invoked with the ``AD_TAF_FLAGS``, which are set in the AD
   options file, by default
   :filelink:`tools/adjoint_options/adjoint_default` (see
   :numref:`adoptfile`). TAF
   writes the resulting AD code into the file ``ad_input_code_ad.f``.

#. A short sed script :filelink:`tools/adjoint_sed <tools/adjoint_sed>` is
   applied to ``ad_input_code_ad.f`` to reinstate :varlink:`myThid` into
   the CALL argument list of the tape I/O routines (``adopen``, ``adread``,
   ``adwrite``, ``adclose``).  The result is written to file
   ``ad_taf_output.f``.

#. If the ``-nocat4ad`` option is specified, the concatenation of all ``.f``
   files is skipped and instead all necessary files are sent to TAF and for
   each file an AD-file is returned.

#. All routines are compiled and an executable is generated.

The ``make ftlall`` target proceeds in the same manner with
#define :varlink:`ALLOW_TANGENTLINEAR_RUN`, the flags ``FTL_TAF_FLAGS`` and
the files ``ftl_input_code.f``, ``ftl_input_code_tl.f`` and
``ftl_taf_output.f``.

.. _ad_taf_flow:

The list ``AD_FLOW_FILES`` and ``.flow`` files
----------------------------------------------

TAMC and TAF can evaluate user-specified directives that start with a
specific syntax (``CADJ``, ``C$TAF``, ``!$TAF``). The main categories of directives
are ``STORE`` directives and ``FLOW`` directives. Here, we are concerned with
flow directives, store directives are treated in
:numref:`ad_taf_checkpointing`.

Flow directives enable the AD tool to evaluate how it should treat
routines that are ’hidden’ by the user, i.e. routines which are not
contained in the ``AD_FILES`` list (see :numref:`sec_ad_hidden`), but which
are called in part of the code that the AD tool does see. The flow
directive tell the AD tool:

-  which subroutine arguments are input/output

-  which subroutine arguments are active

-  which subroutine arguments are required to compute the cost

-  which subroutine arguments are dependent

The syntax for the flow directives can be found in the AD tool manuals.

:filelink:`genmake2 <tools/genmake2>` generates a list (or variable) ``AD_FLOW_FILES``
that contains all files with suffix ``.flow`` that it finds in its search
directories. The flow directives for the core MITgcm routines of
:filelink:`eesupp/src/` and :filelink:`model/src/` reside in :filelink:`pkg/autodiff/`
and in :filelink:`eesupp/src/`. These directories also
contain hand-written adjoint code for the MITgcm WRAPPER (:numref:`wrapper`).

Flow directives for package-specific routines are contained in the
corresponding package directories, generally in a file ``«PKG»_ad.flow``, e.g.,
ptracers-specific directives are in :filelink:`ptracers_ad.flow <pkg/ptracers/ptracers_ad.flow>`.

.. _ad_taf_checkpointing:

Store directives for 3-level checkpointing
------------------------------------------

TAF stores the variables that are named in store directives. These are part
of the MITgcm source code and are enclosed by the CPP option
:varlink:`ALLOW_AUTODIFF_TAMC`. The storing that is required at each period
of the 3-level checkpointing (:numref:`sec_autodiff_storage_v_recompute`)
is controlled by three top-level headers.

::

    do ilev_3 = 1, nchklev_3
    #  include "checkpoint_lev3_directives.h"
       do ilev_2 = 1, nchklev_2
    #     include "checkpoint_lev2_directives.h"
          do ilev_1 = 1, nchklev_1
    #        include "checkpoint_lev1_directives.h"

    ...

          end do
       end do
    end do

All files ``checkpoint_lev?_directives.h`` are contained in directory
:filelink:`pkg/autodiff/`. The lengths of the three loops,
:varlink:`nchklev_1`, :varlink:`nchklev_2` and :varlink:`nchklev_3`, are set
in the header file :filelink:`tamc.h <pkg/autodiff/tamc.h>`; their product
needs to be at least equal to :varlink:`nTimeSteps`.

.. _ad_taf_hooks:

Hand-written adjoint code
-------------------------

The hand-written adjoint and tangent linear routines for TAF
(:numref:`sec_ad_hidden`) reside next to the ``.flow`` files that refer to
them. A flow directive names the routine that TAF calls in place of a
generated one, e.g., for the active read of a 2-D field (file
:filelink:`active_file.flow <pkg/autodiff/active_file.flow>`):

::

    cadj SUBROUTINE active_read_xy FTLNAME = g_active_read_xy
    cadj SUBROUTINE active_read_xy ADNAME  = adactive_read_xy

The hooks (:numref:`ad_output_hooks`) are treated in the same manner.
:filelink:`dummy_in_stepping.F <pkg/autodiff/dummy_in_stepping.F>` is essentially empty, the corresponding adjoint
routine is hand-written rather than generated automatically.
Appropriate flow directives
(:filelink:`dummy_in_stepping.flow <pkg/autodiff/dummy_in_stepping.flow>`)
ensure that
TAF does not automatically generate :filelink:`addummy_in_stepping.F <pkg/autodiff/addummy_in_stepping.F>` by
trying to differentiate :filelink:`dummy_in_stepping.F <pkg/autodiff/dummy_in_stepping.F>`, but instead refers to
the hand-written routine. The hook has no active argument; the directives
``REQUIRED`` and ``INFLUENCED`` ensure that it is called nevertheless, and
its adjoint with it. The flow directives of
:filelink:`dummy_for_etan.flow <pkg/autodiff/dummy_for_etan.flow>` and
:filelink:`autodiff_inadmode.flow <pkg/autodiff/autodiff_inadmode.flow>`
do the same for the other hooks.

:filelink:`dummy_in_stepping.F <pkg/autodiff/dummy_in_stepping.F>` is called in the forward code at the beginning
of each timestep, before the call to :filelink:`model/src/dynamics.F`, thus ensuring that
:filelink:`addummy_in_stepping.F <pkg/autodiff/addummy_in_stepping.F>` is called at the end of each timestep in the
adjoint calculation, after the call to :filelink:`addummy_in_dynamics.F <pkg/autodiff/addummy_in_dynamics.F>`.
The latter is the hand-written adjoint of a further hook, S/R
``DUMMY_IN_DYNAMICS`` in :filelink:`model/src/dynamics.F`, which is used
with TAF only (CPP option :varlink:`ALLOW_AUTODIFF_MONITOR_DIAG`).

:filelink:`addummy_in_stepping.F <pkg/autodiff/addummy_in_stepping.F>`
includes the header files :filelink:`adcommon.h </pkg/autodiff/adcommon.h>`. This
header file is also hand-written. It contains the common blocks of the
adjoint variables, e.g., :varlink:`adDYNVARS_R`, :varlink:`adDYNVARS_CD` and
:varlink:`adDYNVARS_DIFFKR`, which have
been extracted from the adjoint code to enable access to the adjoint
variables.

**WARNING:** If the structure of the common blocks :varlink:`dynvars_r`,
:varlink:`dynvars_cd`, etc., changes similar changes will occur in the
adjoint common blocks. Therefore, consistency between the
TAF-generated common blocks and those in
:filelink:`adcommon.h </pkg/autodiff/adcommon.h>` have to be
checked.

To add a field to the output of the adjoint state
(:numref:`ad_output_dumps_diags`), add its ``DUMP_ADJ_*`` call to
``addummy_in_stepping.F`` (and its adjoint common block to ``adcommon.h`` if
it is not there yet).

.. _sec_autodiff_diva:

Adjoint dump & restart – divided adjoint (DIVA)
-----------------------------------------------

Authors: Patrick Heimbach & Geoffrey Gebbie, 07-Mar-2003

***NOTE:THIS SECTION IS SUBJECT TO CHANGE. IT REFERS TO TAF-1.4.26.**

Old TAF versions are incomplete and have problems with both TAF options
``-pure`` and ``-mpi``. At the time of the latest update, the current version
of TAF is 6.1.5

Introduction
~~~~~~~~~~~~

Most high performance computing (HPC) centers require the use of batch
jobs for code execution. Limits in maximum available CPU time and memory
may prevent the adjoint code execution from fitting into any of the
available queues. This presents a serious limit for large scale / long
time adjoint ocean and climate model integrations. The MITgcm itself
enables the split of the total model integration into sub-intervals
through standard dump/restart of/from the full model state. For a
similar procedure to run in reverse mode, the adjoint model requires, in
addition to the model state, the adjoint model state, i.e., all variables
with derivative information which are needed in an adjoint restart. This
adjoint dump & restart is also termed ’divided adjoint (DIVA)’.

For this to work in conjunction with automatic differentiation, an AD
tool needs to perform the following tasks:

#. identify an adjoint state, i.e., those sensitivities whose
   accumulation is interrupted by a dump/restart and which influence the
   outcome of the gradient. Ideally, this state consists of

   -  the adjoint of the model state,

   -  the adjoint of other intermediate results (such as control
      variables, cost function contributions, etc.)

   -  bookkeeping indices (such as loop indices, etc.)

#. generate code for storing and reading adjoint state variables

#. generate code for bookkeeping , i.e., maintaining a file with index
   information

#. generate a suitable adjoint loop to propagate adjoint values for
   dump/restart with a minimum overhead of adjoint intermediate values.

TAF (but not TAMC!) generates adjoint code which performs the above
specified tasks. It is closely tied to the adjoint multi-level
checkpointing. The adjoint state is dumped (and restarted) at each step
of the outermost checkpointing level and adjoint integration is
performed over one outermost checkpointing interval. Prior to the
adjoint computations, a full forward sweep is performed to generate the
outermost (forward state) tapes and to calculate the cost function. In
the current implementation, the forward sweep is immediately followed by
the first adjoint leg. Thus, in theory, the following steps are
performed (automatically)

-  **1st model call:**
   This is the case if file ``costfinal`` does *not* exist. S/R
   ``mdthe_main_loop.f`` (generated by TAF) is called.

   #. calculate forward trajectory and dump model state after each
      outermost checkpointing interval to files ``tapelev3``

   #. calculate cost function ``fc`` and write it to file ``costfinal``

-  **2nd and all remaining model calls:**
   This is the case if file costfinal *does* exist. S/R
   ``adthe_main_loop.f`` (generated by TAF) is called.

   #. (forward run and cost function call is avoided since all values
      are known)

      -  if 1st adjoint leg:
         create index file ``divided.ctrl`` which contains info on current
         checkpointing index :math:`ilev3`

      -  if not :math:`i`-th adjoint leg:
         adjoint picks up at :math:`ilev3 = nlev3-i+1` and runs to
         :math:`nlev3 - i`

   #. perform adjoint leg from :math:`nlev3-i+1` to :math:`nlev3 - i`

   #. dump adjoint state to file ``snapshot``

   #. dump index file ``divided.ctrl`` for next adjoint leg

   #. in the last step the gradient is written.

A few modifications were performed in the forward code, obvious ones
such as adding the corresponding TAF-directive at the appropriate place,
and less obvious ones (avoid some re-initializations, when in an
intermediate adjoint integration interval).

[For TAF-1.4.20 a number of hand-modifications were necessary to
compensate for TAF bugs. Since we refer to TAF-1.4.26 onwards, these
modifications are not documented here].

.. _diva_recipe:

Recipe for divided adjoint code generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verification experiment :filelink:`lab_sea <verification/lab_sea>` tests the
divided adjoint and serves as an example of how to configure the code.

#. define ``USE_DIVA=1``, either as an environment variable (e.g., in bash:
   ``export USE_DIVA=1``), in a ``genmake_local`` file in the ``build``
   directory, or in your build options file. This will instruct
   :filelink:`genmake2 <tools/genmake2>` to generate TAF options (``-pure``)
   for divided adjoint generation.

#. In a local copy of :filelink:`AUTODIFF_OPTIONS.h
   <pkg/autodiff/AUTODIFF_OPTIONS.h>` set:

   - #define :varlink:`ALLOW_DIVIDED_ADJOINT`

   to enable code for divided adjoint.

#. If using MPI, make sure that the paths to mpi-header files, such as
   ``mpif.h``, are know to :filelink:`genmake2 <tools/genmake2>` (as usual, via
   the build options file, see also :numref:`diva_mpi`).

#. Run the usual sequence for generating the Makefile and the AD-code.

   ::

      ../../../tools/genmake2  -mods=../code_ad -nocat4ad [ other options ]
      make depend
      make adtaf

   the ``-nocat4ad`` option is not necessary, but will generate individual
   AD-files for each forward file sent to TAF. The adjoint code now contains
   subroutines (in ``the_main_loop_ad.f``):

   -  ``mdthe_main_loop``:
      Is responsible for the forward trajectory, storing of outermost
      checkpoint levels to file, computation of cost function, and
      storing of cost function to file (1st step).

   -  ``adthe_main_loop``:
      Is responsible for computing one adjoint leg, dump adjoint state
      to file and write index info to file (2nd and consecutive
      steps).

   Then compile with ``make adall`` (the ``make adtaf`` step is not necessary
   unless you want to inspect the TAF-generated code before compiling).

.. _diva_mpi:

Special considerations for multi processor (MPI) runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the machine where you execute the code (most likely not the machine where
you run TAF) find the includes directory for MPI containing ``mpif.h``. Either
copy ``mpif.h`` to the machine where you preprocess the code (generate the
``.f`` files) before TAF-ing, or add the path to the includes directory to your
:filelink:`genmake2 <tools/genmake2>` platform setup. TAF needs some MPI
parameter settings (essentially ``mpi_comm_world`` and ``mpi_integer``) to
incorporate those in the adjoint code. The ``-mpi`` will be added to the TAF
argument list automatically.

.. _ad_tapenade:

Adjoint code generation using Tapenade
======================================

Please refer to Gaikwad et al. (2024) :cite:`gaikwad:24` for more details and a comparative analysis with TAF. Recently, introduction of the profiling capabilities in Tapenade have resulted in substantial insights and speedups for the Tapenade-generated adjoint, see Hascoet et al. (2024) :cite:`hascoet:24`.

Feel free to reach out if you wish to use Tapenade and need help!

Authors: Shreyas Sunil Gaikwad, Sri Hari Krishna Naryanan, Laurent Hascoet, Patrick
Heimbach

Introduction
------------

TAPENADE is an open-source Automatic Differentiation Engine developed at INRIA
Sophia-Antipolis by the Tropics then Ecuador teams. TAPENADE can be utilized as
a server (JAVA servlet), which runs at INRIA Sophia-Antipolis. The current
address of this TAPENADE server is `here
<http://www-tapenade.inria.fr:8080/tapenade/index.jsp>`_. TAPENADE can also be
downloaded and installed locally as a set of JAVA classes (JAR archive). In
that case it is run by a simple command line, which can be included into a
Makefile. It also provides you with a user-interface to visualize the results
in a HTML browser.

Downloading and installing Tapenade
-----------------------------------

While the MITgcm source files are prepared to generate adjoint sensitivities,
they will not be able to do so without an operable installation of
Tapenade. Fortunately the Tapenade installation procedure is straight forward.

We detail the instructions here, but the latest instructions can always be
found `here
<https://tapenade.gitlabpages.inria.fr/tapenade/distrib/README.html>`__.

Prerequisites for Linux or Mac OS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before installing Tapenade, you must check that an up-to-date Java Runtime
Environment is installed. Tapenade will not run with older Java Runtime
Environment.

Steps for Mac OS
~~~~~~~~~~~~~~~~

Tapenade 3.16 distribution does not contain a fortranParser executable
for MacOS. You need docker on your Mac to run the Tapenade
distribution with Fortran programs with a docker image from `here
<https://gitlab.inria.fr/tapenade/tapenade>`__. Details on how to
build your own fortranParser is `here
<https://tapenade.gitlabpages.inria.fr/tapenade/docs/html/src/frontf/README.html?highlight=mac>`__.
You may also build Tapenade on your Mac from the `gitlab repository
<https://tapenade.gitlabpages.inria.fr/tapenade/docs/html/distrib/README.html>`__.

Running a docker image requires absolute paths, e.g., to
:filelink:`tools/TAP_support/flow_tap <tools/TAP_support/flow_tap>`.
To make it work,

1. use the option ``-rootdir`` at the :filelink:`genmake2
   <tools/genmake2>` step, or alternatively export environment
   variable ``MITGCM_ROOTDIR``, to specify the absolute path to your
   MITgcm directory (see also :numref:`command_line_options`).

2. bind mount the absolute path in the docker command as a volume by putting
   ::

      BASEDIR="$(cd "$(dirname "$0")" && cd ../ && pwd)"
      TAPENADECMD="docker container run --rm -u $(stat -f '%u:%g' ./) \
                -v \${PWD}:\${PWD} -v ${BASEDIR}:${BASEDIR} -w \${PWD} \
	        registry.gitlab.inria.fr/tapenade/tapenade"

   in your build-options or in a ``genmake_local`` file
   (:numref:`genmake2_desc`). ``BASEDIR`` should expand to your
   root directory (check ``TAPENADECMD`` in ``Makefile``).

In order to run :filelink:`./testreport -tap $moreoption
<verification/testreport>` in :filelink:`verification <verification>`,
the root directory can be passed to :filelink:`genmake2
<tools/genmake2>` via ``export MITGCM_ROOTDIR=$BASEDIR`` or setting it
in your built-options or ``genmake_local`` file.

Steps for Linux
~~~~~~~~~~~~~~~

1. Read `the Tapenade license. <https://tapenade.gitlabpages.inria.fr/userdoc/build/html/LICENSE.html>`__

2. Download `tapenade_3.16.tar
   <https://tapenade.gitlabpages.inria.fr/tapenade/distrib/tapenade_3.16.tar>`__
   into your chosen installation directory *install_dir*.

3. Go to your chosen installation directory *install_dir*, and extract Tapenade
   from the tar file :

::

    % tar xvfz tapenade_3.16.tar

4. On Linux, depending on your distribution, Tapenade may require you to set
   the shell variable ``JAVA_HOME`` to your java installation directory. It is
   often ``JAVA_HOME=/usr/java/default``. You might also need to modify the
   ``PATH`` by adding the bin directory from the Tapenade installation. An
   example can be found :ref:`here <tapenade_bashrc_snippet>`.

Prerequisites for Windows
~~~~~~~~~~~~~~~~~~~~~~~~~

Before installing Tapenade, you must check that an up-to-date Java Runtime
Environment is installed. Tapenade will not run with older Java Runtime
Environment. The Fortran parser of Tapenade uses `cygwin
<https://www.cygwin.com/>`__.

Steps for Windows
~~~~~~~~~~~~~~~~~

1. Read `the Tapenade license. <https://tapenade.gitlabpages.inria.fr/userdoc/build/html/LICENSE.html>`__

2. Download `tapenade_3.16.zip
   <https://tapenade.gitlabpages.inria.fr/tapenade/distrib/tapenade_3.16.zip>`__
   into your chosen installation directory *install_dir*.

3. Go to your chosen installation directory *install_dir*, and extract Tapenade
   from the zip file.

4. Save a copy of the ``install_dir\tapenade_3.16\bin\tapenade.bat`` file and
   modify ``install_dir\tapenade_3.16\bin\tapenade.bat`` according to your
   installation parameters:

replace ``TAPENADE_HOME=..`` by ``TAPENADE_HOME="install_dir"\tapenade_3.16``
replace ``JAVA_HOME="C:\Progra~1\Java\jdkXXXX"`` by your current java directory
replace ``BROWSER="C:\Program Files\Internet Explorer\iexplore.exe"`` by your
current browser.

.. _tapenade_bashrc_snippet:

**NOTE**: Every time you wish to use the AD capability with Tapenade, you must re-source the environment. We recommend that this be done automatically in your bash or c-shell profile upon login. An example of an addition to a ``.bashrc`` file from a Linux server is given below. Luckily, shell variable ``JAVA_HOME`` was not required to be explicitly set for this particular Linux distribution, but might be necessary for some other distributions.

::

    ##set some env variables for tapenade

    export TAPENADE_HOME="/home/shreyas/tapenade_3.16"
    export PATH="$PATH:$TAPENADE_HOME/bin"

    ##Modules

    module use /share/modulefiles/
    module load java/jdk/16.0.1 # Java required by Tapenade

You should now have a working copy of Tapenade.

For more information on the tapenade command and its arguments, type :

::

    tapenade -?

Prerequisites for Tapenade setup
--------------------------------

The ``packages.conf`` file should include both the ``adjoint`` and ``tapenade``
packages. Not all packages have been tested with Tapenade. The users are
referred to the ``code_tap`` directories in the various
verification experiments for reference.

**Pro tip**: ``diff -qr dir1 dir2`` can help you see all the differences in the files of two directories.

``autodiff`` is not completely untangled from the Tapenade setup yet. In
``code_tap/AUTODIFF_OPTIONS.h``, the only flags that can be defined safely are
``ALLOW_AUTODIFF_MONITOR`` (which enables the adjoint output described in
:numref:`ad_output`) and ``AUTODIFF_ALLOW_VISCFACADJ``
(:numref:`ad_inadmode_switches`).

Rest of the setup remains unchanged.

.. _ad_tap_build:

Building MITgcm TLM with Tapenade
---------------------------------

The setup remains similar to how one sets up the TLM with TAF. A typical flow
will look as follows -

::

    ### Assuming $PWD is the build subdirectory
    ### Clean stuff
    make CLEAN

    ### Use your own optfile
    ../../../tools/genmake2 -tap -of ../../../tools/build_options/linux_amd64_ifort -mods ../code_tap
    make depend

    ### Differentiate code to generate TLM code using Tapenade
    ### Creates executable mitgcmuv_tap_tlm
    make -j 8 tap_tlm

    ### Rest of the setup is standard
    cd ../run
    rm -r *
    ln -s ../input_tap/* .
    ./prepare_run
    ln -s ../build/mitgcmuv_tap_tlm .
    ./mitgcmuv_tap_tlm > output_tap_tlm.txt 2>&1

Note that with Tapenade the tangent linear model is called from the gradient
check package only (:numref:`ad_gradient_check`), so that
:varlink:`useGrdchk` ``= .TRUE.`` is required in ``data.pkg``.

Building MITgcm adjoint with Tapenade
-------------------------------------

The setup remains similar to how one sets up the adjoint with TAF. A typical
flow will look as follows -

::

    ### Assuming $PWD is the build subdirectory
    ### Clean stuff
    make CLEAN

    ### Use your own optfile
    ../../../tools/genmake2 -tap -of ../../../tools/build_options/linux_amd64_ifort -mods ../code_tap
    make depend

    ### Differentiate code to generate adjoint code using Tapenade
    ### Creates executable mitgcmuv_tap_adj
    make -j 8 tap_adj

    ### Rest of the setup is standard
    ### These commands are for a typical verification experiment
    cd ../run
    rm -r *
    ln -s ../input_tap/* .
    ./prepare_run
    ln -s ../build/mitgcmuv_tap_adj .
    ./mitgcmuv_tap_adj > output_tap_adj.txt 2>&1

.. _ad_tap_build_detail:

The AD build process in detail
------------------------------

For Tapenade, the ``make tap_adj`` target consists of the following
procedures (cf. :numref:`sec_ad_build`):

#. A header file ``AD_CONFIG.h`` is generated which contains the CPP option
   #define :varlink:`ALLOW_ADJOINT_RUN`.

#. Tapenade is invoked in reverse mode (``tapenade -b``) with the
   ``TAPENADE_FLAGS`` for all ``.f`` files that are part of the list
   ``AD_FILES``. The default flags are set in
   :filelink:`tools/adjoint_options/adjoint_tap` (see :numref:`adoptfile`).
   They name the top-level routine with its dependent and independent
   variables, and the external library file
   :filelink:`tools/TAP_support/flow_tap` (:numref:`ad_tap_handwritten`).

#. For each source file ``«NAME».f`` Tapenade writes the adjoint code into
   a file ``«NAME»_b.f``. If ``-nocat4ad`` is not specified, these files are
   concatenated into a single file ``adj_tap_all.f``.

#. All routines are compiled and an executable is generated.

The ``make tap_tlm`` target proceeds in the same manner with
#define :varlink:`ALLOW_TANGENTLINEAR_RUN`, Tapenade in tangent mode
(``tapenade -d``), and the files ``«NAME»_d.f`` and ``tlm_tap_all.f``.

Tapenade requires no store directives (:numref:`sec_ad_storing`). It
determines by itself which intermediate results the reverse sweep requires
and keeps them on a stack (:filelink:`adStack.c <pkg/tapenade/adStack.c>`).
By default Tapenade checkpoints every subroutine call: in the reverse sweep
a routine is executed once more to record its intermediate results before
its adjoint is executed. A routine that is differentiated in split mode
(Tapenade option ``-nocheckpoint``) records them in the forward sweep
instead; in this case Tapenade generates the two routines ``«NAME»_fwd``
and ``«NAME»_bwd`` in place of ``«NAME»_b``.

.. _ad_tap_handwritten:

Hand-written derivative code
----------------------------

Tapenade has no counterpart of the flow directives of TAF
(:numref:`ad_taf_flow`). The routines that are hidden from it
(:numref:`sec_ad_hidden`) are described in its external library file
:filelink:`tools/TAP_support/flow_tap`. For each argument of a routine an
entry of this file states whether the argument is read, written, or both,
e.g., for the active read of a 2-D field:

::

    subroutine active_read_xy:
      external:
      shape: (param 1, param 2, param 3, param 4, param 5, param 6, param 7, param 8)
      type: (none(),none(),none(),none(),none(),none(),none(),none())
      NotReadNotWritten:  (0,0,0,0,0,0,0,0)
      ReadNotWritten:     (1,0,1,1,1,1,1,1)
      NotReadThenWritten: (0,1,0,0,0,0,0,0)
      ReadThenWritten:    (0,0,0,0,0,0,0,0)

For a call of such a routine with an active argument Tapenade generates a
call to a routine of the same name with the suffix ``_B`` (adjoint), ``_D``
(tangent-linear) or ``_FWD``/``_BWD`` (adjoint without checkpointing), with
the derivative of each active argument as an additional argument. These
routines are written by hand. Most of them reside in
:filelink:`pkg/tapenade`, e.g., those of the active file routines in
:filelink:`active_read_tap.F <pkg/tapenade/active_read_tap.F>` and
:filelink:`active_write_tap.F <pkg/tapenade/active_write_tap.F>`; those of
the exchange routines reside in :filelink:`eesupp/src` and
:filelink:`pkg/exch2`.

The external library describes data flow only, and a call whose arguments
are all passive is left out of the generated code. Each hook
(:numref:`ad_output_hooks`) therefore has a Tapenade counterpart, called
beside it under ``ALLOW_TAPENADE``. S/R ``DUMMY_FOR_ETAN_TAP``,
``AUTODIFF_INADMODE_SET_TAP`` and ``AUTODIFF_INADMODE_UNSET_TAP`` (file
:filelink:`dummy_tap.F <pkg/tapenade/dummy_tap.F>`) take the field they act
on as an argument, declared in ``flow_tap`` as read and written. Tapenade
then generates, at the mirror image of the forward call, a call to the
hook's ``_B``, ``_D`` or ``_FWD``/``_BWD`` routine, with the derivative of
the field as an additional argument. These routines are written by hand in
``dummy_tap.F`` and call the same ``DUMP_ADJ_*``, ``WRITE_FLD_*`` and
``ADAUTODIFF_INADMODE_*`` routines as the TAF path; the adjoint state
arrives as an argument, so no equivalent of ``adcommon.h`` is needed. The
counterpart of S/R ``DUMMY_IN_STEPPING``, S/R ``DUMMY_IN_STEPPING_TAP``
(file
:filelink:`dummy_in_stepping_tap.F <pkg/tapenade/dummy_in_stepping_tap.F>`),
is the one routine of the package that Tapenade differentiates (listed in
:filelink:`tapenade_ad_diff.list <pkg/tapenade/tapenade_ad_diff.list>`). It
passes the state and forcing fields one per call to hooks of the same kind,
so that a field which is passive in a given configuration loses only its
own call and the argument list of every generated call stays fixed. The mode
switches take the 2-D ``etaN`` as their argument only so that Tapenade
generates the call; its adjoint is not modified.

To add a field to the output of the adjoint state
(:numref:`ad_output_dumps_diags`), add one guarded call in
``dummy_in_stepping_tap.F`` and, for a field shape without a hook, a hook
family in ``dummy_tap.F`` with its ``flow_tap`` stanza.

.. _ad_openad:

Adjoint code generation using OpenAD
====================================

**IMPORTANT NOTE:** As OpenAD is no longer maintained (latest OpenAD snapshot
at Argonne National Lab was from March 2014), MITgcm stopped supporting the
OpenAD interface after the ``checkpoint69o`` tag (from July 2026). In case you
need to use OpenAD for a specific application, please use the last OpenAD
supported code:

::

    % git checkout checkpoint69o

Authors: Jean Utke, Patrick Heimbach and Chris Hill

Introduction
------------

The development of OpenAD was initiated as part of the ACTS (Adjoint
Compiler Technology & Standards) project funded by the NSF Information
Technology Research (ITR) program. The main goals for OpenAD initially
defined for the ACTS project are:

#. develop a flexible, modular, open source tool that can generate
   adjoint codes of numerical simulation programs,

#. establish a platform for easy implementation and testing of source
   transformation algorithms via a language-independent abstract
   intermediate representation,

#. support for source code written in C and Fortan, and

#. generate efficient tangent linear and adjoint for the MIT general
   circulation model.

OpenAD’s homepage is at http://www-unix.mcs.anl.gov/OpenAD. A
development WIKI is at
http://wiki.mcs.anl.gov/OpenAD/index.php/Main_Page. From the WIKI’s
main page, click on `Handling GCM <https://wiki.mcs.anl.gov/OpenAD/index.php/Handling_GCM>`_
for various aspects pertaining to
differentiating the MITgcm with OpenAD.

Downloading and installing OpenAD
---------------------------------

The OpenAD webpage has a detailed description on how to download and
build OpenAD. From its homepage, please click on
`Binaries <http://www.mcs.anl.gov/OpenAD/binaries.shtml>`_. You may either download pre-built binaries
for quick trial, or follow the detailed build process described at
http://www.mcs.anl.gov/OpenAD/access.shtml.

Building MITgcm adjoint with OpenAD
-----------------------------------

**17-January-2008**

OpenAD was successfully built on head node of ``itrda.acesgrid.org``,
for following system:

::

    > uname -a
    Linux itrda 2.6.22.2-42.fc6 #1 SMP Wed Aug 15 12:34:26 EDT 2007 i686 i686 i386 GNU/Linux

    > cat /proc/version
    Linux version 2.6.22.2-42.fc6 (brewbuilder@hs20-bc2-4.build.redhat.com)
    (gcc version 4.1.2 20070626 (Red Hat 4.1.2-13)) #1 SMP Wed Aug 15 12:34:26 EDT 2007

    > module load ifc/9.1.036 icc/9.1.042

Head of MITgcm branch (``checkpoint59m`` with some modifications) was used for
building adjoint code. Following routing needed special care (revert
to revision 1.1): http://wwwcvs.mitgcm.org/viewvc/MITgcm/MITgcm_contrib/heimbach/OpenAD/OAD_support/active_module.f90?hideattic=0&view=markup.

Building the MITgcm adjoint using an OpenAD Singularity container
-----------------------------------------------------------------

The MITgcm adjoint can also be built using a Singularity container.  You will
need `Singularity <https://singularity.hpcng.org/>`_, version 3.X.  A container
with OpenAD can be downloaded from the Sylabs Cloud: [#thanks-Dan]_

::

   singularity pull library://jahn/default/openad:latest

To use it, supply the path to the downloaded container to genmake2,

::

   ../../../tools/genmake2 -oad -oadsingularity /path/to/openad_latest.sif ...
   make adAll

If your build directory is on a remotely mounted file system (mounted at
/mountpoint), you may have to add an option for mounting it in the container:

::

   ../../../tools/genmake2 -oad -oadsngl "-B /mountpoint /path/to/openad_latest.sif" ...

The ``-oadsingularity`` option is also supported by testreport,
:numref:`testreport_utility`.  Note that the path to the container has to be
either absolute or relative to the build directory.

.. rubric:: Footnotes

.. [#thanks-Dan] A big thank you to Dan Goldberg for supplying the definition
   file for the Singularity container!
