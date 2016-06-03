package = "rltorch"
version = "scm-1"

source = {
   url = "git://github.com/torch/denoyer/rl.git",
}

description = {
   summary = "Deep Sequential Neural Network package for Torchsf ",
   detailed = [[
   ]],
   homepage = "https://github.com/torch/denoyer/rqsdqsdqsdl",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && echo $(PREFIX) && $(MAKE) install"
}
