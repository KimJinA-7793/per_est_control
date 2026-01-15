from setuptools import setup, find_packages
import os

package_name = 'vision_estimation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            [os.path.join('resource', package_name)]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join(
                'share', package_name,
                'vision_estimation', 'perception', 'resources'
            ),
            ['vision_estimation/perception/resources/camcalib.npz'],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='gina7793@gmail.com',
    description='vision + estimation package',
    license='TODO',
    entry_points={
        'console_scripts': [
            'vies = vision_estimation.node:main',
        ],
    },
)