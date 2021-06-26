# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

packages = \
['msckf_tutorial']

package_data = \
{'': ['*']}

install_requires = \
['click>=8.0.1,<9.0.0',
 'moderngl-window>=2.4.0,<3.0.0',
 'moderngl>=5.6.4,<6.0.0',
 'numpy>=1.21.0,<2.0.0',
 'opencv-python>=4.5.2,<5.0.0',
 'pyrr>=0.10.3,<0.11.0',
 'scipy>=1.7.0,<2.0.0',
 'transforms3d>=0.3.1,<0.4.0']

entry_points = \
{'console_scripts': ['format = scripts.scripts:format',
                     'generate_setup_py = scripts.scripts:generate_setup_py']}

setup_kwargs = {
    'name': 'msckf-tutorial',
    'version': '0.1.0',
    'description': '',
    'long_description': None,
    'author': 'Edwinem',
    'author_email': '735010+Edwinem@users.noreply.github.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.8,<3.10',
}


setup(**setup_kwargs)

