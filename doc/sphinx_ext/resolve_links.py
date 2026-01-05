from urllib.parse import quote
import inspect
import logging
from importlib import import_module

_logger = logging.getLogger("fastsurfer.doc.linkcode_resolve")

class LinkCodeResolver:

    def __init__(self, url, branch="dev"):
        self._gh_url = url
        self._branch = branch

    @property
    def base_path(self):
        # Base URL to the GitHub repository where the source code is hosted
        return f"{self._gh_url}/blob/{self._branch}"

    def __call__(self, domain, info):
        # Check if the domain is Python, if not return None
        if domain != "py":
            return None
        if not info["module"]:
            return None

        # Import the module using the module information
        mod = import_module(info["module"])

        # Check if the fullname contains a ".", indicating it's a method or attribute of a class
        if "." in info["fullname"]:
            objname, attrname = info["fullname"].split(".")
            # Get the object from the module
            _mod = getattr(mod, objname)
            try:
                # Try to get the attribute from the object
                obj = getattr(_mod, attrname)
            except AttributeError:
                # If the attribute doesn't exist, return None
                return None
        else:
            # If the fullname doesn't contain a ".", get the object directly from the module
            obj = getattr(mod, info["fullname"])
            _mod = mod

        # If the object is a property, try to link to the getter or setter method
        if isinstance(obj, property):
            obj = obj.fget or obj.fset
        try:
            # Try to get the source file and line numbers of the object
            lines, first_line = inspect.getsourcelines(obj)
        except (TypeError, OSError) as e:
            lines, first_line = [], -1
            # if the obj is a variable, try to find it in the parent object
            try:
                _lines, _first_line = inspect.getsourcelines(_mod)
                from re import match, escape
                # try to find the line this was defined in
                for idx, line in enumerate(_lines):
                    if match(f"\\s*{escape(info['fullname'])}\\s*[:=]", line):
                        first_line = _first_line + idx
                        lines = [_lines[idx]]
                        break
            except (OSError, TypeError):
                # if this was unsuccessful, give up
                pass

            if first_line == -1 and isinstance(e, OSError):
                # If the object is not a Python object that has a source file, return None
                _suffix = "" if str(obj).startswith("<") else f" ({obj})"
                _logger.info(f"Could not find source for {info['module']}.{info['fullname']}{_suffix}.")
                return None

        # Replace "." with "/" in the module name to construct the file path
        filename = quote(info["module"].replace(".", "/"))
        # If the filename doesn't start with "tests", add a "/" at the beginning
        if not filename.startswith("tests"):
            filename = "/" + filename

        # Construct the URL that points to the source code of the object on GitHub
        return f"{self.base_path}{filename}.py#L{first_line}-L{first_line + len(lines) - 1}"
