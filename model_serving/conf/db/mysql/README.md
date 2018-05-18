# palette关系数据库说明

* palette_users 用户表，用于存储用户的基本信息
* palette_user_groups 用户群组表，用于存储用户间的组关系，用于用户组管理
* palette_user_authorization 用户授权表，用于记录用户的权限信息，该授权表记录用户与service表、device表、pets表中记录的关系
* palette_services 服务表，用于管理系统的服务信息
* palette_scenes 场景表，用于管理平台提供的场景服务，数量估计不多，但是考虑到后续可能的扩展以及后端管理系统的接入，因此用专门的SQL表进行管理
* palette_pets 宠物表，用于支持后续如果真的有用户使用时，对宠物信息进行管理
* palette_devices 设备表，用户支持后续如果真的有用户使用时，对用户的设备信息进行管理
* palette_medias 媒体表，用于存储每次用于检测而上传的图片或者视频的文件信息以及对应的检测结果



