## Beta reconstruction

The aim of this project is to write a Python package to reconstruct high temperature beta (cubic) phases from alpha (HCP) phases measured with EBSD at low temperature.

For the materials we consider, they are primarily alpha (HCP) crystals at room temperature and primarily beta (cubic) crystals at high temperature (on the order of 1000 degrees C). Heating from low to high temperature causes a phase transformation from alpha to beta. Upon cooling, needle like crystals of alpha are precipitated from the beta phase. Most of the time, all of the beta changes back to alpha on cooling but sometimes some beta is retained to room temperature. Some of the time not all of the alpha is converted to beta upon heating… this retained alpha phase is called primary alpha. All of this is dependent on the heating protocol. As well as indexed alpha and beta phases the dataset may have some unindexed points where the Euler angles are not known.

There are a number of possible orientations for each alpha or beta crystal. The beta crystals are hard to measure directly since 1000 degress is really quite hot, however you can work out the orientation of the beta crystals from looking at the orientation of the alpha crystals since there is a relation between the alpha and beta orientations. The complication is that there is more than one possible beta orientation for each alpha orientation. You can narrow down the options (usually to a single unique orientation) by looking at multiple neighbouring grains.

The approximate workflow for reconstruction is:

* Read in data – any combination of alpha and beta and unindexed points
* Dilate the alpha phase to fill the whole map (In the simplest scenario we use only the alpha data to reconstruct the beta)
* Identify the grains (variants) – Identify the grains via a flood fill based on a boundary condition
* Beta reconstruction from the identified variants

Some code was previously written by Peter Davies of Sheffield University. Although it works its performance is limited the code is hard to extend. We want to replicate and extend the original functionality in Python.

We already have some of the above functions in DefDap (https://github.com/MechMicroMan/DefDAP) so will use the existing code wherever possible. 
