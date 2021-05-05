from core.utils.lr_scheduler import get_scheduler_by_name

fun = get_scheduler_by_name("mnist")

# print(fun)

def test(a=1):
    print(a)


test(b=2)