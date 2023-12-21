# Training data

This page provides technical and demographic information about the training data that was used for the FastSurferVINN project.
For details, also refer to the corresponding [publication](https://doi.org/10.1016/j.neuroimage.2022.118933).

## Demographic information

study | population | min | max | median | mean | q15 | q85 | female | total | region
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
study | population | min | max | median | mean | q15 | q85 | female | total | region
ABIDE-I | ASD/Normal | 18 | 46 | 24 | 26 | 20 | 32 | 34 | 68 | Europe, NorthAmerica
ADNI | AD/MCI/Normal | 56 | 91 | 73 | 73 | 66 | 80 | 106 | 215 | NorthAmerica
HCP | Normal | 24 | 33 | 28 | 28 | 24 | 33 | 15 | 30 | NorthAmerica
IXI | Normal | 20 | 86 | 51 | 50 | 28 | 68 | 204 | 400 | Europe
LA5C | Neuropsych/Normal | 21 | 50 | 32 | 33 | 23 | 46 | 102 | 203 | NorthAmerica
MBB | Normal | 56 | 86 | 71 | 70 | 62 | 76 | 15 | 30 | Europe
MIRIAD | AD/Normal | 23 | 78 | 33 | 42 | 23 | 68 | 96 | 195 | Europe
OASIS1 | Normal | 18 | 90 | 58 | 54 | 26 | 75 | 39 | 79 | NorthAmerica
OASIS2 | AD/Normal | 60 | 90 | 75 | 75 | 67 | 83 | 32 | 65 | NorthAmerica
RS | Normal | 30 | 95 | 69 | 59 | 42 | 76 | 15 | 30 | Europe

All statistics refer to the "big" (N=1315) training sample. q15 and q85 refer to the 15th and 85th percentile of the distribution.


## Technical information

study | vendor | model | fieldstrength | n | voxel size | sequence
--  | --  | --  | --  | --  | --  | --  
abide-i | GE | mixed | 3T | 2 | 1.0 mm | mixed
abide-i | Philips | mixed | 3T | 7 | 1.0 mm | mixed
abide-i | Siemens | mixed | 3T | 59 | 1.0 mm | mixed
adni | GE | mixed | 1.5T | 35 | 1.0 mm | IR-SPGR
adni | GE | mixed | 3T | 35 | 1.0 mm | IR-SPGR
adni | Other | mixed | 3T | 1 | 1.0 mm | N/A
adni | Philips | mixed | 1.5T | 1 | 1.0 mm | MPRAGE
adni | Philips | mixed | 3T | 23 | 1.0 mm | MPRAGE
adni | Siemens | mixed | 1.5T | 30 | 1.0 mm | MPRAGE
adni | Siemens | mixed | 3T | 90 | 1.0 mm | MPRAGE
hcp | Siemens | Skyra | 3T | 30 | 0.7 mm | MEMPRAGE
ixi | GE | N/A | 1.5T | 50 | 1.0 mm | N/A
ixi | Philips | Intera | 1.5T | 225 | 1.0 mm | N/A
ixi | Philips | Intera | 3T | 125 | 1.0 mm | N/A
la5c | Siemens | Trio | 3T | 203 | 1.0 mm | MPRAGE
miriad | GE | Signa | 1.5T | 30 | 1.0 mm | IR-FSPGR
mpi-mbb | Siemens | Verio | 3T | 195 | 1.0 mm | MP2RAGE
oasis1 | Siemens | Vision | 1.5T | 79 | 1.0 mm | MPRAGE
oasis2 | Siemens | Vision | 1.5T | 65 | 1.0 mm | MPRAGE
rs | Siemens | Prisma | 3T | 30 | 0.8 mm | MEMPRAGE, MPRAGE

'Mixed' refers to multi-center studies with heterogeneous acquisition devices, where detailed information must be obtained from the respective data repositores. 'N/A' means not available.


## References and repositories

study | reference | url
--  | --  | --  
abide-i | Di Martino et al. (2013) | https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html
adni | Mueller et al. (2005) | https://www.adni-info.org
hcp | Van Essen et al. (2012) | https://www.humanconnectome.org
ixi | N/A | https://brain-development.org/ixi-dataset
la5c | Poldrack et al. (2016)  | https://openfmri.org/dataset/ds000030
miriad | Malone et al. (2013) | http://miriad.drc.ion.ucl.ac.uk
mbb | Mendes et al. (2019) | https://openfmri.org/dataset/ds000221
oasis1 | Marcus et al.(2007) | https://www.oasis-brains.org
oasis2 | Marcus et al.(2010) | https://www.oasis-brains.org
rs | Breteler et al. (2014) | N/A

'N/A' means not available.


## Bibliography

- M.M. Breteler, T. Stöcker, E. Pracht, D. Brenner, R. Stirnberg. MRI in the Rhineland Study: a novel protocol for population neuroimaging. Alzheimer’s Dementia, 10 (4) (2014), p. P92

- A. Di Martino, C. Yan, Q. Li, E. Denio, F. Castellanos, K. Alaerts, J. Anderson, M. Assaf, S. Bookheimer, M. Dapretto, et al. The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. Mol. Psychiatry, 19 (2013), pp. 659-667. https://doi.org/10.1038%2Fmp.2013.78

- I.B. Malone, D. Cash, G.R. Ridgway, D.G. MacManus, S. Ourselin, N.C. Fox, J.M. Schott. Miriad-public release of a multiple time point Alzheimer’s MR imaging dataset. NeuroImage, 70 (2013), pp. 33-36. https://doi.org/10.1016/j.neuroimage.2012.12.044

- D.S. Marcus, A.F. Fotenos, J.G. Csernansky, J.C. Morris, R.L. Buckner. Open access series of imaging studies: longitudinal MRI data in nondemented and demented older adults. J. Cogn. Neurosci., 22 (12) (2010), pp. 2677-2684. https://doi.org/10.1162/jocn.2009.21407

- D.S. Marcus, T.H. Wang, J. Parker, J.G. Csernansky, J.C. Morris, R.L. Buckner. Open access series of imaging studies (OASIS): cross-sectional MRI data in young, middle aged, nondemented, and demented older adults. J. Cogn. Neurosci., 19 (9) (2007), pp. 1498-1507. https://doi.org/10.1162/jocn.2007.19.9.1498

- N. Mendes, S. Oligschläger, M.E. Lauckner, J. Golchert, J.M. Huntenburg, M. Falkiewicz, M. Ellamil, S. Krause, B.M. Baczkowski, R. Cozatl, et al. A functional connectome phenotyping dataset including cognitive state and personality measures. Sci. Data, 6 (1) (2019), p. 180307. https://doi.org/10.1038/sdata.2018.307

- S.G. Mueller, M.W. Weiner, L.J. Thal, R.C. Petersen, C.R. Jack, W. Jagust, J.Q. Trojanowski, A.W. Toga, L. Beckett. Ways toward an early diagnosis in Alzheimer’s disease: the Alzheimer’s disease neuroimaging initiative (ADNI). Alzheimer’s Dementia, 1 (1) (2005), pp. 55-66. https://doi.org/10.1016/j.jalz.2005.06.003

- R.A. Poldrack, E. Congdon, W. Triplett, K. Gorgolewski, K. Karlsgodt, J. Mumford, F. Sabb, N. Freimer, E. London, T. Cannon, et al. A phenome-wide examination of neural and cognitive function. Sci. Data, 3 (2016), p. 160110. https://doi.org/10.1038/sdata.2016.110

- D.C. Van Essen, K. Ugurbil, E. Auerbach, D. Barch, T. Behrens, R. Bucholz, A. Chang, L. Chen, M. Corbetta, S.W. Curtiss, et al. The human connectome project: a data acquisition perspective. NeuroImage, 62 (4) (2012), pp. 2222-2231. https://doi.org/10.1016/j.neuroimage.2012.02.018
