from setuptools import find_packages, setup

package_name = 'my_tb4_ex'

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
    maintainer='kim',
    maintainer_email='jongun1203@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'nav_to_pose_docking = my_tb4_ex.nav_to_pose_docking:main',
            'nav_to_pose_undock = my_tb4_ex.nav_to_pose_undock:main',
            'target_pose_pub = my_tb4_ex.target_pose_pub:main',
            'yolo_test = my_tb4_ex.yolo_test:main',
            'sim_time_test = my_tb4_ex.sim_time_test:main',
            'follow_pid = my_tb4_ex.follow_pid:main',
            'yolo_cam_img_pub = my_tb4_ex.yolo_cam_img_pub:main',
        ],
    },
)
