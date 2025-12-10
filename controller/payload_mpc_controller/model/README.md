# How to compile the MPC code generation

1. Install ACADO as described in http://acado.github.io/install_linux.html

2. Source the file ACADO_ROOT/build/acdo_env.sh

3. Go into payload_mpc_controller/model

4. Prepare the make with "cmake ."

5. Compile it with "make"
   
6. Modify the parameters in "config/model.yaml" to match your model

7. Run the executable to generate all code "./quadrotor_payload_mpc_with_ext_force_codegen ../config/model.yaml" 
   
   (You can specify the model parameter file path)

Thanks to [rpg_mpc](https://github.com/uzh-rpg/rpg_mpc)


# 如何编译MPC代码生成

1. 按照 http://acado.github.io/install_linux.html 的说明安装ACADO

2. 执行文件 ACADO_ROOT/build/acdo_env.sh

3. 进入 payload_mpc_controller/model 目录

4. 使用 "cmake ." 准备编译

5. 使用 "make" 进行编译
   
6. 修改 "config/model.yaml" 中的参数以匹配您的模型

7. 运行可执行文件生成所有代码 "./quadrotor_payload_mpc_with_ext_force_codegen ../config/model.yaml" 
   
   （您可以指定模型参数文件的路径）

感谢 [rpg_mpc](https://github.com/uzh-rpg/rpg_mpc)
