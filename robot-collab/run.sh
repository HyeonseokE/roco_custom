# LD_PRELOAD=/home/hscho/anaconda3/envs/roco/x86_64-conda-linux-gnu/sysroot/usr/lib64/libGL.so.1 python run_dialog.py --task pack --llm gpt-4o

# "sort": SortOneBlockTask,
# "cabinet": CabinetTask,
# "rope": MoveRopeTask,
# "sweep": SweepTask,
# "sandwich": MakeSandwichTask,
# "pack": PackGroceryTask,

# vglrun python run_dialog.py --task sweep --llm gpt-5-nano
# vglrun python run_dialog.py --task sweep --llm gpt-4o
vglrun python run_dialog.py --task sweep --llm gemini-2.5-flash

# 5회 수행.
# vglrun python run_dialog.py --task sweep --llm gpt-4o --skip_display
# vglrun python run_dialog.py --task sweep --llm gpt-4o --skip_display
# vglrun python run_dialog.py --task sweep --llm gpt-4o --skip_display
# vglrun python run_dialog.py --task sweep --llm gpt-4o --skip_display

# vglrun python run_dialog.py --task sandwich --llm gpt-4o --skip_display
# vglrun python run_dialog.py --task sandwich --llm gpt-4o --skip_display
# vglrun python run_dialog.py --task sandwich --llm gpt-4o --skip_display
# vglrun python run_dialog.py --task sandwich --llm gpt-4o --skip_display
# vglrun python run_dialog.py --task sandwich --llm gpt-4o --skip_display

# vglrun python run_dialog.py --task pack --llm gpt-4o --skip_display
# vglrun python run_dialog.py --task pack --llm gpt-4o --skip_display
# vglrun python run_dialog.py --task pack --llm gpt-4o --skip_display
# vglrun python run_dialog.py --task pack --llm gpt-4o --skip_display
# vglrun python run_dialog.py --task pack --llm gpt-4o --skip_display