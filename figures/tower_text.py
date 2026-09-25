# pylint: disable=too-many-arguments,too-many-positional-arguments
"""Tower names in their tower colour (REF / OPT1 / OPT2).

``colored_label`` splits a label on the tower names and draws each token in
its colour, so fold labels such as ``REF+OPT1 -> OPT2`` read at a glance.
"""
import re

from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker

from figures import paper_style as ps

# REF text uses the legibility grey: #b8b8b8 washes out as type.
TEXT_COLOR = {"REF": ps.REF_DARK, "OPT1": ps.OPT1, "OPT2": ps.OPT2}
_SPLIT = re.compile(r"(REF|OPT1|OPT2)")


def tower_color(name):
    """Text colour of one tower name (case-insensitive)."""
    return TEXT_COLOR[name.upper()]


def _tokens(line, default):
    return [
        (tok, TEXT_COLOR.get(tok, default)) for tok in _SPLIT.split(line) if tok
    ]


def colored_box(label, fontsize, default=ps.AXIS_INK, rotation=0):
    """Offsetbox with every tower name of ``label`` in its colour.

    Lines are separated by ``\\n``; ``rotation`` is 0 or 90.
    """
    lines = []
    for line in label.split("\n"):
        parts = [
            TextArea(tok,
                     textprops={
                         "color": col,
                         "fontsize": fontsize,
                         "rotation": rotation
                     }) for tok, col in _tokens(line, default)
        ]
        if rotation == 90:
            lines.append(
                VPacker(children=parts[::-1], align="center", pad=0, sep=0))
        else:
            lines.append(HPacker(children=parts, align="baseline", pad=0,
                                 sep=0))
    if rotation == 90:
        return HPacker(children=lines, align="center", pad=0, sep=1)
    return VPacker(children=lines, align="center", pad=0, sep=1)


def colored_label(ax,
                  label,
                  xy,
                  xycoords="axes fraction",
                  fontsize=None,
                  default=ps.AXIS_INK,
                  rotation=0,
                  box_alignment=(0.5, 0.5)):
    """Place a multi-colour ``label`` on ``ax`` at ``xy``."""
    box = colored_box(label, fontsize or ps.LABEL_SIZE, default, rotation)
    abox = AnnotationBbox(box,
                          xy,
                          xycoords=xycoords,
                          frameon=False,
                          box_alignment=box_alignment,
                          pad=0,
                          annotation_clip=False)
    ax.add_artist(abox)
    return abox
