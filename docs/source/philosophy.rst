Philosophy
==========

Motivation
----------
As the availability of datasets required for groundwater modeling has increased, so too has the need for
improved methods for rapidly assessing stresses to aquatic systems such as cumulative pumping
impacts. Analytic element groundwater flow models such as GFLOW provide an efficient, rapid, and
flexible method for evaluating drawdown and surface water depletion at multiple scales. However, the
most common approach to constructing analytic element models involves manually digitizing important
hydrologic features (e.g. streams, rivers, lakes), using existing hydrographic and topographic layers as
guides. Although typically less cumbersome than building finite difference grids, this process can still be
prohibitively labor-intensive for many water resources decisions. 


What linesink-maker does
------------------------------
Linesink-maker leverages existing information in the National Hydrography Dataset (NHDPlus) to automate construction of line-sink elements representing streams, rivers, and lakes. 

* Vector information defining the waterbody geometries is read in and simplified using a line simplification algorithm in the Shapely Python package. The level of simplification is controlled by a distance tolerance, which can be varied by geographic area (allowing for different levels of complexity in the model near- and farfields). Further simplification can be achieved using ancillary information in NHDPlus such as stream order and arbolate sum. Other properties, such as resistance, width and parameter group, can also be specified. 
* A prototyping mode facilitates selection of an optimal distance tolerance to achieve a desired number of line-sink equations. 
* The program produces a linesink file for import into the GFLOW graphical user interface. 
* For a demonstration of how linesink-maker works, see the Nicolet example, which constructs a 3,000+ equation stream network in a matter of minutes.

What linesink-maker doesn't do
------------------------------
Linesink-maker does not construct a complete, working GFLOW model. It is still on the modeler to specify reasonable recharge rates and hydraulic conductivity values, including any heterogeniety that needs to be represented. More importantly, the modeler must also compare the model output to observed values of stream (base) flow and hydraulic head, and estimated reasonable values for hydraulic conductivity and recharge. GFLOW makes this easy to do with its PEST integration.
