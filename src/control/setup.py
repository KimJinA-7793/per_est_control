from setuptools import find_packages, setup

package_name = 'control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eipih',
    maintainer_email='2001sonickim@gmail.comm',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'recycle = control.recycle:main',
            'tower = control.main:main',
            'control_client = control.nodes.control_client:main',
            'per_est_server_demo = control.nodes.per_est_server_demo:main',
            'control_client_demo = control.nodes.control_client_demo:main',
            'control_runner = control.nodes.control_runner:main',

        ],
    },
)
