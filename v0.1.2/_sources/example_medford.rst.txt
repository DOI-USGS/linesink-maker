Medford National Forest Unit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this example, a `GFLOW <https://www.haitjema.com/>`_ model of a National Forest Unit in Northern Wisconsin is constructed, similar to the one documented in `this study <https://wgnhs.wisc.edu/pubs/tr0041/>`_. The files for this example can be found in the `examples/medford subfolder <https://github.com/aleaf/linesink-maker/tree/develop/examples/medford>`_ of the Linesink-maker repository.

Configuration file in yaml format (``Medford_lines.yml``)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. literalinclude:: ../../examples/medford/Medford_lines.yml
    :language: yaml
    :linenos:

The ``GlobalSettings:`` block in the configuration file contains settings that apply to the whole model. The ``resistance:`` , ``global_streambed_thickness:``, ``H:`` (representative aquifer thickness), and ``k:`` (representative aquifer hydraulic conductivity) are all used to compute the characteristic leakage length (:math:`\lambda`) needed for estimating an appropriate width parameter for lakes (Haitjema, 2005), and are given in the model length units (``ComputationalUnits:``). ``working_dir:`` specifies the location where output from Linesink-maker will be written; ``prj:`` is a file path to a projection file containing a `well-known text (WKT) <http://docs.opengeospatial.org/is/18-010r7/18-010r7.html>`_ representation of the projected CRS for the model.

The ``ModelDomain:`` block allows specification of different areas of model refinement. With the nearfield: key, the user can specify a polygon shapefile defining the primary area of focus for the model. With the optional routed_area: key, another polygon can be supplied to define the extent of the model stream network of routed, resistance linesinks (see Haitjema, 1995). Finally, the outer extent of the model can be defined using a polygon shapefile with the ``farfield:`` key, or alternatively, as a buffer distance around the nearfield polygon with the ``farfield_buffer:`` key. The area between the nearfield polygon (or optionally, the routed area polygon) and the farfield extent is then populated with zero-resistance linesinks that form a perimeter boundary condition (see Haitjema, 1995). 

Source hydrography input are defined in the ``NHDFiles:`` block as shown. Currently, Linesink-maker only works with NHDPlus data. The ``Simplification:`` block controls how the hydrography input are discretized, and which features are retained. For example, a ``nearfield_tolerance:`` value of 100 meters means that the simplification of the original flowlines will be limited by the constraint that the simplified lines do not deviate from the original lines by more than this distance. With the ``min_farfield_order:`` key, lower-order streams can be excluded from the farfield linesinks (a value of 2 means that first-order streams are excluded). The ``min_waterbody_size:``, ``min_nearfield_wb_size:`` and ``min_farfield_wb_size:`` keys control the minimimum size for the waterbodies that are included in the routed, nearfield and farfield areas of the model (in square km). Finally, with the ``drop_intermittent:`` key, streams classified as “intermittent” in NHDPlus can be excluded from the routed part of the model outside of the model nearfield. By default, all streams are included in the nearfield.

The above configuration file can be used in the following script to generate a linesink string (LSS) XML file of the stream network that can be imported into `GFLOW <https://www.haitjema.com/>`_ (version 2.2 or later). 

Python script to build the model (``make_linesinks.py``):
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. literalinclude:: ../../examples/medford/make_linesinks.py
    :language: python
    :linenos:

.. note::
    The ``input_file`` path given in the above script assumes that the script is being executed in the same folder as `Medford_lines.yml`.

A shapefile representation of the linesinks is also produced, along with additional shapefiles of the source hydrography merged and clipped to the model area. The resulting linesinks are shown in Figure 1.


.. figure:: medford_results.png
    :align: left
    :scale: 65 %
    :alt: alternate text

    Figure 1: Linesinks produced by Linesink-maker for the Medford Unit of the Chequamegon-Nicolet National Forest. A distance tolerance between the simplified linesinks and original hydrography controls the level of detail in the stream network. Linesinks within the forest unit were created at the highlest level of detail (100 meter distance tolerance). A buffer of routed resistance linesinks disretized at a 300 meter tolerance surrounds the forest unit, to allow for accurate simulation of hydraulic divides between competing sinks and stream (base) flows into the forest unit. Coarsely discretized (500 meter tolerance) zero-resistance linesinks create a perimeter boundary condition for the solution. B illustrates the conversion of flowlines (red) into a drainage lake represented by routed resistance linesinks around its perimeter. Linesink-maker makes small adjustments to the end elevations of drainage lake tributaries to ensure proper routing in the GFLOW GUI.



Alternative XML format for configuration file (``Medford_lines.xml``)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. literalinclude:: ../../examples/medford/Medford_lines.xml
    :language: xml
    :linenos:
