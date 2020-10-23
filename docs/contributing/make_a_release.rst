==============
Make a Release
==============

We did implement different CI/CD pipelines with GitHub Actions. This eases the
release process but a small checklist is always a great help.


#. Update the version

    .. code-block:: bash

        # make sure you are on the right branch
        git checkout develop

        # update the version.txt to the targeted version
        git add version.txt
        git commit -m "update version.txt"
        git push origin develop

#. Add a tag to the commit

    .. code-block:: bash

        git tag vx.y
        git push origin vx.y

#. Release

    .. code-block:: bash

        git checkout master
        git merge develop

    * Make the `GitHub Release <https://github.com/timeatlas-dev/timeatlas/releases/new>`_.

#. Check if everything went well

    * On `GitHub Actions <https://github.com/timeatlas-dev/timeatlas/actions>`_.
    * On `PyPI <https://pypi.org/project/timeatlas/>`_.
    
