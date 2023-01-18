from IPython.extensions.autoreload import AutoreloadMagics


def autoreload():
    AutoreloadMagics().autoreload("3")
