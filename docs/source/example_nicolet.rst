Nicolet National Forest Unit (Northern part)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This page shows another example of using Linesink-maker to construct a `GFLOW <https://www.haitjema.com/>`_ model, for `the northern part of the Nicolet National Forest Unit in Northern Wisconsin <https://wgnhs.wisc.edu/pubs/tr0042/>`_. The files for this example can be found in the `examples/nicolet subfolder <https://github.com/aleaf/linesink-maker/tree/develop/examples/nicolet>`_ of the Linesink-maker repository. To run the example, simply run the included ``make_linesinks.py`` script at the command line with ``Nicolet_lines.xml`` configuration file as an argument:

.. code-block::

    python make_linesinks.py Nicolet_lines.xml

See the :ref:`Medford example <Medford National Forest Unit>` for a more detailed description of the input options, and an example of the more human-readable YAML configuration file format.

Configuration file in XML format (``Nicolet_lines.xml``)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. literalinclude:: ../../examples/nicolet/Nicolet_lines.xml
    :language: xml
    :linenos:

``make_linesinks.py``
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. literalinclude:: ../../examples/nicolet/make_linesinks.py
    :language: python
    :linenos: