// Point the links in the announcement bar at the sibling documentation tree.
//
// The bar is one HTML string shared by every page, so a relative path would resolve differently
// depending on how deep the page is. Every page loads its assets from _static at the tree root, so
// resolving any of those gives the root, and the sibling tree is one level up from there. That
// keeps this independent of the domain, of the path prefix the site is served under, and of any
// theme or config option: without _static there would be no stylesheet and no page to read.
window.addEventListener("DOMContentLoaded", function () {
    var asset = document.querySelector('link[href*="_static/"], script[src*="_static/"]');
    if (!asset) {
        return;  // nothing to anchor on, leave the fallback href in place
    }
    var url = new URL(asset.getAttribute("href") || asset.getAttribute("src"), window.location.href);
    var root = new URL(url.pathname.slice(0, url.pathname.lastIndexOf("_static/")), url);
    document.querySelectorAll("a[data-doc-tree]").forEach(function (link) {
        link.href = new URL("../" + link.dataset.docTree + "/", root).href;
    });
});
