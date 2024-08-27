# Contributing to FastSurfer

All types of contributions are encouraged and valued. The community looks forward to your contributions.

## Reporting Bugs

### Before Submitting a Bug Report

Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment components/versions.
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](https://github.com/Deep-MI/FastSurfer/issues?q=label%3Abug).
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Version of the interpreter, compiler, SDK, runtime environment, package manager, depending on what seems relevant.
  - Possibly your input and the output.
  - Can you reliably reproduce the issue? And can you also reproduce it with older versions?

### How Do I Submit a Good Bug Report?

We use GitHub issues to track bugs and errors. If you run into an issue with the project:

- Open an [Issue](https://github.com/Deep-MI/FastSurfer/issues/new). (Since we can't be sure at this point whether it is a bug or not, we ask you not to talk about a bug yet and not to label the issue.)
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the *reproduction steps* that someone else can follow to recreate the issue on their own. This usually includes your code. For good bug reports you should isolate the problem and create a reduced test case.
- Provide the information you collected in the previous section.
- Also provide the $subjid/scripts/recon-surf.log (if existent) and in the case of a parallel run, also the $subjid/scripts/[l/r]h.processing.cmdf.log (if existent).

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps. If there are no reproduction steps or no obvious way to reproduce the issue, the team will ask you for those steps and mark the issue as `needs-repro`. Bugs with the `needs-repro` tag will not be addressed until they are reproduced.
- If the team is able to reproduce the issue, it will be marked `needs-fix`, as well as possibly other tags (such as `critical`).

## Suggesting Enhancements

Please follow these guidelines to help maintainers and the community to understand your suggestion for enhancements.

### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the documentation carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](https://github.com/Deep-MI/FastSurfer/issues) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on/plugin library.

### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://github.com/Deep-MI/FastSurfer/issues).

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many details as possible.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
- **Explain why this enhancement would be useful** to most users.

## Contributing Code

1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) this repository to your github account
2. Clone your fork to your computer (`git clone https://github.com/<username>/FastSurfer.git`)
3. Change into the project directory (`cd FastSurfer`)
4. Add Deep-MI repo as upstream (`git remote add upstream https://github.com/Deep-MI/FastSurfer.git`)
5. Update information from upstream (`git fetch upstream`)
6. Checkout the upstream dev branch (`git checkout -b dev upstream/dev`)
7. Create your feature branch from dev (`git checkout -b my-new-feature`)
8. Commit your changes (`git commit -am 'Add some feature'`)
9. Push to the branch to your github (`git push origin my-new-feature`)
10. Create new pull request on github web interface from that branch into Deep-NI **dev branch** (not into stable)

If lots of things changed in the meantime or the pull request is showing conflicts you should rebase your branch to the current upstream dev.
This is the preferred way, but only possible if you are the sole develop or your branch:

10. Switch into dev branch (`git checkout dev`)
11. Update your dev branch (`git pull upstream dev`)
12. Switch into your feature (`git checkout my-new-feature`)
13. Rebase your branch onto dev (`git rebase dev`), resolve conflicts and continue until complete
14. Force push the updated feature branch to your gihub (`git push -f origin my-new-feature`)

If other people co-develop the my-new-feature branch, rewriting history with a rebase is not possible.
Instead you need to merge upstream dev into your branch:

10. Switch into dev branch (`git checkout dev`)
11. Update your dev branch (`git pull upstream dev`)
12. Switch into your feature (`git checkout my-new-feature`)
13. Merge dev into your feature (`git merge dev`), resolve conflicts and commit
14. Push to origin (`git push origin my-new-feature`)

Either method updates the pull request and resolves conflicts, so that we can merge it once it is complete.
Once the pull request is merged by us you can delete the feature branch in your clone and on your fork:

15. Switch into dev branch (`git checkout dev`)
16. Delete feature branch (`git branch -D my-new-feature`)
17. Delete the branch on your github fork either via GUI, or via command line (`git push origin --delete my-new-feature`)

This procedure will ensure that your local dev branch always follows our dev branch and will never diverge. You can, once in a while, push the dev branch, or similarly update stable and push it to your fork (origin), but that is not really necessary. 

Next time you contribute a feature, you do not need to go through the steps 1-6 above, but simply:
- Switch to dev branch (`git checkout dev`)
- Make sure it is identical to upstream (`git pull upstream dev`)
- Check out a new feature branch and continue from 7. above. 

Another good command, if for some reasons your dev branch diverged, which should never happen as you never commit to it, you can reset it by `git reset --hard upstream/dev`. Make absolutely sure you are in your dev branch (not the feature branch) and be aware that this will delete any local changes!

## Attribution

This guide is based on the **contributing-gen**. [Make your own](https://github.com/bttger/contributing-gen)!
