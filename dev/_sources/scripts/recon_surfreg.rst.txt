Surface registration: recon-surfreg.sh
======================================

``recon-surfreg.sh`` adds the spherical registration (``?h.sphere`` and ``?h.sphere.reg``) to a subject that was
processed with ``--no_surfreg``. The registration to FreeSurfer's ``fsaverage`` is needed for any cross-subject
analysis of thickness maps, so this lets you add it later without running the surface reconstruction again. It needs
a FreeSurfer license, like the surface reconstruction.

Usage:

.. code-block:: text

    <fastsurfer_home>/recon_surf/recon-surfreg.sh --sd <subjects_dir> \
        --sid <subject_id> --fs_license <freesurfer_license_path>

Full commandline interface of recon-surfreg.sh
----------------------------------------------

.. command-output:: ./recon_surf/recon-surfreg.sh --help
   :cwd: /../
