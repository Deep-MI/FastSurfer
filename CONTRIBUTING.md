Contribution Guide
==================
All types of contributions are encouraged and valued. The community looks forward to your contributions.

Reporting Bugs
--------------

### Before Submitting a Bug Report
Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if you your setup is supported, in particular, check if you are using the correct versions of python
  packages, FreeSurfer, the operating system and drivers. You may use `$FASTSURFER_HOME/run_fastsurfer.sh --version all`
  to get a list of versions of the python packages used by FastSurfer.
- Search the [Issue Tracker](https://github.com/Deep-MI/FastSurfer/issues?q=is%3Aissue) to see if other users have 
  previously experienced the same issue or similar issues to yours. You may find a solution in a response to an issue
  reported by another user.
- Collect information about your issue:
  - Error messages and stack traces (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - FastSurfer installation type: native, Docker, Singularity
  - Versions of relevant libraries, in particular those reported by `$FASTSURFER_HOME/run_fastsurfer.sh --version all`
  - How did you run FastSurfer (command and arguments)? 
  - Your input and output files
  - Can you reliably reproduce the issue (e.g. on other computers)? And can you also reproduce it with older versions?

### How Do I Submit a Good Bug Report?
We use GitHub issues to track bugs and errors. If you run into an issue with the project:

- Open an [Issue in GitHub](https://github.com/Deep-MI/FastSurfer/issues/new). 
- Select the Issue type that best describes the Issue type you are reporting. "Bugs" describe situations, where the 
  observed and the intended behavior are different. If you are about the intended behavior, it is best to report the 
  Issue as "Question/Help/Support" and ask about the intended behavior or "Documentation".
- Closely follow the template format presented to you and answer the questions asked.  
- In particular, please explain the behavior you would expect and the observed behavior.
- Please provide as much context as possible and describe the *reproduction steps* that someone else can follow to 
  recreate the issue on their own. This usually includes your code. For good bug reports you should isolate the problem 
  and create a reduced test case.
- Provide the information you collected in the previous section.
- Also provide log files and screenshots as indicated in the template, e.g. `<subject_dir>/scripts/deep-seg.log`, 
  `<subject_dir>/scripts/recon-surf.log`, and `<subject_dir>/scripts/[l/r]h.processing.cmdf.log` (if these exist).

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps. If there are no reproduction steps or no 
  obvious way to reproduce the issue, the team will ask you for those steps and mark the issue as `needs-repro`. Bugs 
  with the `needs-repro` tag will not be addressed until they are reproduced.
- If the team is able to reproduce the issue, it will be marked `needs-fix`, as well as possibly other tags (such as 
  `critical`).

Suggesting Enhancements
-----------------------
Please follow these guidelines to help maintainers and the community to understand your suggestion for enhancements.

### Before Submitting an Enhancement
- Make sure that you are using the latest version.
- Read the documentation carefully and find out if the functionality is already covered, maybe by an individual 
  configuration.
- Perform a [search](https://github.com/Deep-MI/FastSurfer/issues?q=is%3Aissue) to see if the enhancement has already  
  been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to 
  convince the project's developers of the merits of this feature. Keep in mind that we want features that will be 
  useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, 
  consider writing an add-on/plugin library.

### How Do I Submit a Good Enhancement Suggestion?
Enhancement suggestions are tracked as [GitHub issues](https://github.com/Deep-MI/FastSurfer/issues).

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many details as possible.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
- **Explain why this enhancement would be useful** to most users.

Contributing Code
-----------------
1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the 
   [official FastSurfer](https://github.com/Deep-MI/FastSurfer) repository to your GitHub account.
2. Clone your fork to your computer: `git clone https://github.com/<username>/FastSurfer.git`
3. Change into the project directory: `cd FastSurfer`
4. Add Deep-MI repo as upstream: `git remote add upstream https://github.com/Deep-MI/FastSurfer.git`
5. Update information from upstream: `git fetch upstream`
6. Checkout the upstream dev branch: `git checkout -b dev upstream/dev`
7. Create your feature branch from dev: `git checkout -b my-new-feature`

   ```bash
   git clone https://github.com/<username>/FastSurfer.git
   cd FastSurfer
   git remote add upstream https://github.com/Deep-MI/FastSurfer.git
   git fetch upstream
   git checkout -b dev upstream/dev
   git checkout -b my-new-feature
   ```

8. Edit the code and implement your changes/features 
9. Commit your changes: `git commit -am 'Add some feature'`
10. Push to the branch to your GitHub: `git push origin my-new-feature`

   ```bash
   git commit -am 'Add some feature'
   git push origin my-new-feature
   ```

11. [Create a new pull request on the GitHub web interface](https://github.com/Deep-MI/FastSurfer/compare) from your 
    branch into Deep-NI **dev branch** (not into stable): Select "Compare across forks" and then your fork on the right,
    finally, select your `my-new-feature` branch to compare to.

If lots of things changed on the official FastSurfer repository in the meantime or the pull request is showing
conflicts, these need to be resolved by either rebasing (preferred) or merging.
    
### Option 1: Rebasing
Rebasing is preferred, because it leaves a cleaner history of changes. However, rebasing is only possible if you are the
sole developer collaborating on your branch. To rebase, do the following:

12. Switch into dev branch: `git checkout dev`
13. Update your dev branch: `git pull upstream dev`
14. Switch into your feature: `git checkout my-new-feature`
15. Rebase your branch onto dev, resolve conflicts and continue until complete:  `git rebase dev`
16. Force push the updated feature branch to your GitHub: `git push -f origin my-new-feature`

    ```bash
    git checkout dev
    git pull upstream dev
    git checkout my-new-feature
    git rebase dev
    git push -f origin my-new-feature
    ```

### Option 2: Merging
If other people co-develop in the `my-new-feature` branch, rewriting history with a rebase is not possible.
Instead, you need to merge `upstream/dev` into your branch:

12. Switch into dev branch: `git checkout dev`
13. Update your dev branch: `git pull upstream dev`
14. Switch into your feature: `git checkout my-new-feature`
15. Merge dev into your feature, resolve conflicts and commit: `git merge dev`
16. Push to origin: `git push origin my-new-feature`

    ```bash
    git checkout dev
    git pull upstream dev
    git checkout my-new-feature
    git merge dev
    git push origin my-new-feature
    ```

Either method updates the pull request and resolves conflicts, so that we can merge it once it is complete.
Once the pull request is merged by us you can delete the feature branch locally and on your fork:

17. Switch into dev branch: `git checkout dev`
18. Delete feature branch: `git branch -D my-new-feature`
19. Delete the branch on your GitHub fork either via GUI, or via command line: `git push origin --delete my-new-feature`

This procedure will ensure that your local `dev` branch always follows our `dev` branch and will never diverge. You can,
once in a while, push the `dev` branch, or similarly update `stable` and push it to your fork (`origin`), but that is 
not really necessary. 

Next time you contribute a feature, steps 1-6 above are simpler as you already have a local FastSurfer copy. Simply:
- Switch to dev branch: `git checkout dev`
- Make sure it is identical to upstream: `git pull upstream dev`
- Check out a new feature branch and continue from 7. above. 

Another good command, if -- for any reason -- your `dev` branch diverged, which should never happen as you never 
commit to it, you can reset it by `git reset --hard upstream/dev`. Make absolutely sure you are in your `dev` branch 
(not the feature branch) and be aware that this will delete any local changes!

Attribution
-----------
This guide is based on the **contributing-gen**. [Make your own](https://github.com/bttger/contributing-gen)!
