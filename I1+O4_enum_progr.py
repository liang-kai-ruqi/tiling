import numpy
import time

'''
版本介绍：
该程序用于中等规模的矩形区域内的王氏铺陈的计数。

状态用ndarray保存，不支持稀疏张量。
支持在屏幕上更改砖组系数。支持负系数。不支持在屏幕上直接更改其他参数，其他参数需在程序内修改。
模数表为从上界开始向下选取的素数。
各个位置横，纵字符表大小可以不相同。
结果追加在文本文件后，格式为f"{c}x{r}: {N}"。注释以"# "开头。
不支持砖集运算。
'''

progr_lastTime = [time.time() for _ in range(4)]
'''上次显示进度的时间'''


def show_progr(text="", progr: int | tuple | float = 0, index=0, startTime=None):
    """
    在终端上显示当前的进度
    :param text: 描述
    :param progr: 进度
    :param index: 该进度的序号（决定了文本的缩进）
    :param startTime: 进程的开始时间，用于预测完成的时间
    """
    global progr_lastTime

    if time.time() - progr_lastTime[index] < progr_delay:  # 未超过延迟时间，不更新进度
        return

    if type(progr) is float:  # 进度是单个浮点数
        text += f"进度{str(100 * progr)[:4]}%"
    elif type(progr) is tuple:  # 进度是两个数的比
        text += f"进度{progr[0]}/{progr[1]}={str(100 * progr[0] / progr[1])[:4]}%"
        progr = progr[0] / progr[1]

    # 填了开始时间参数，则预测剩余时间
    if startTime is not None and progr > 0:
        spentTime = time.time() - startTime
        text += f"，用时{int(spentTime)}s，预计还需{int(spentTime / progr - spentTime)}s(/{int(spentTime / progr)}s)"

    print("# " + index * "  " + "…" + text)
    progr_lastTime[index] = time.time()


def mod_Prod(mat1, Mat2, mod):
    """模乘法，右乘多个矩阵后拼接"""
    """这是本程序中的决速步，当前的写法是已知最优的"""
    # 重整mat1，使其列数等于mat2的行数，相乘后取模
    mat1 = mat1.reshape((-1, Mat2.shape[1]))
    mat1 = mat1 @ Mat2
    mat1 %= mod

    return mat1


def euclid(a, b):
    """辗转相除, ax+by=c"""
    if a == 0:
        return b, 0, 1
    else:
        c, x, y = euclid(b % a, a)
        x_, y_ = y - (b // a) * x, x
        return c, x_, y_


def renew_prime_list(T, T_x0=None):
    """根据砖集张量表示的最大列和更新素数表"""
    global mod_list

    if dtype == numpy.uint32 or dtype == numpy.uint16:
        max_colSum = numpy.max(numpy.sum(T, axis=1))
        if T_x0 is not None:
            max_colSum = max(max_colSum, numpy.max(numpy.sum(T_x0, axis=1)))
    elif dtype == numpy.int32 or dtype == numpy.int16:
        T_pos = numpy.copy(T)
        T_neg = numpy.copy(T)
        for (b, pa, q) in numpy.ndindex(T.shape):
            if T_pos[b, pa, q] < 0:
                T_pos[b, pa, q] = 0
            if T_neg[b, pa, q] > 0:
                T_neg[b, pa, q] = 0
        max_colSum = max(numpy.max(numpy.sum(T_pos, axis=1)), -numpy.min(numpy.sum(T_neg, axis=1)))

        if T_x0 is not None:
            T_pos = numpy.copy(T_x0)
            T_neg = numpy.copy(T_x0)
            for (b, pa, q) in numpy.ndindex(T_x0.shape):
                if T_pos[b, pa, q] < 0:
                    T_pos[b, pa, q] = 0
                if T_neg[b, pa, q] > 0:
                    T_neg[b, pa, q] = 0
            max_colSum = max(numpy.max(numpy.sum(T_pos, axis=1)), -numpy.min(numpy.sum(T_neg, axis=1)), max_colSum)
    else:
        raise NotImplemented

    mod_list = []
    startTime = time.time()

    # 上界默认为能当前列和下保证矩阵乘法不会溢出
    if dtype == numpy.uint32 or dtype == numpy.uint16:
        upbound = 2 ** (dsize * 8) // max(max_colSum, 1) + 1
    elif dtype == numpy.int32 or dtype == numpy.int16:
        upbound = (2 ** (dsize * 7) - 1) // max(max_colSum, 1) + 1
    else:
        raise NotImplemented

    # 从上界倒序遍历寻找素数
    for n in range(upbound, 1, -1):
        is_prime = True  # n是否为素数

        # 从2到sqrt(n)搜索因子，有因子则不是素数
        for i in range(2, int(numpy.sqrt(n)) + 1):
            if n % i == 0:
                is_prime = False
                break

        # 是素数则将其加入列表
        if is_prime:
            mod_list.append(n)

        # 列表足够长则停止寻找素数
        if modNum is not None and len(mod_list) >= modNum:
            break
        else:
            show_progr("正在生成素数表：", (len(mod_list), modNum), 1, startTime=startTime)
    # 记录乘积的log10
    prod_log10 = 0
    for p in mod_list:
        prod_log10 += numpy.log10(p)

    print(f"# 已根据最大列和={max_colSum}，上界={upbound}生成素数表\n# prime_list={mod_list}")
    print(f"# 共{len(mod_list)}个素数，积为{str(10 ** (prod_log10 - int(prod_log10)))[:6]}*10^{int(prod_log10)}，"
          + f"用时{time.time() - startTime}s")


def til_tiles(W, T_list, mod):
    """
    指定砖列的铺陈，输出下边缘。
    :param W: 输入的状态张量
    :param T_list: 转移张量的列表，按从左到右顺序包含整行
    :param mod: 模数
    :return: 输出的状态张量
    """
    for i, tiles in enumerate(T_list):
        W = mod_Prod(W, tiles, mod)
    return W


def til_enum(ini_W, T_list: list, width: int):
    """
    给出宽度和张量列，进行铺陈并计数
    :param ini_W: 初始的状态张量，一般取作[[1]]
    :param T_list: 转移张量列表，按顺序遍历螺旋柱面上所有的位置
    :param width: 宽度，用于自动计算高度，以及决定什么时候输出W_(0..0)的值
    """
    global mod_list, modNum

    enum_startTime = time.time()  # 初始时间

    max_height = len(T_list) // width  # 铺陈最大高度（层数）

    Rem = [0 for _ in range(max_height + 1)]  # 累计的模计数，为Rem[height]=Rem
    Rem_unchange = [32000] + [0 for _ in range(max_height)]  # 记录该位置的Rem未发生改变的次数，第一位总是记为32000
    Mod = 1  # 累计的模

    now_modNum = 0 # 已经选取了的素数个数
    last_height = 0 # 上一次计算到了的高度
    for mod in mod_list:
        now_modNum += 1  # 当前使用了的素数计数

        # 逐高度进行铺陈，更新rem
        tile_startTime = time.time()
        rem = [0]  # 当前素数的模计数，为rem[height]=rem, rem[0]留空
        W = numpy.copy(ini_W)  # 过程序列，初始为初始序列
        for height in range(1, max_height + 1):
            W = til_tiles(W, T_list[(height - 1) * width:height * width], mod)
            W = W.reshape((-1,))
            rem.append(int(W[0]))  # 0位置就是当前层封闭铺陈的计数，将其加入记录

            show_progr(
                f"正在计算宽度width={width_bias[0]}+{width_bias[1]}x{width}，第{now_modNum}个素数，第{height}层铺陈：",
                (height, max_height),
                index=1, startTime=tile_startTime)

        _, Inv, inv = euclid(Mod, mod)  # 取Mod*Inv+mod*inv=1
        Mod_, Rem_ = mod * Mod, 0 # 新的Mod, Rem, 后者初始值为0

        # 记录各高度模计数（列表）连续不变的次数，有停机机制
        # --------------------------------------------------
        shred = int(16 / numpy.log10(mod)) + 1  # 要求犯错概率小于10^(-16)

        now_height = last_height  # 当前不变次数达到要求的高度。last_height是上一次的达到要求的高度。
        now_height_0 = last_height  # 当前不变次数大于0的高度。仅用于估算进度。
        changed = False  # 之前是否已经有高度发生了变化

        # 遍历所有高度更新Rem[height]，并记录每个高度未发生变化的次数
        for height in range(last_height + 1, max_height + 1):
            Rem_ = (rem[height] * Mod * Inv + Rem[height] * mod * inv) % Mod_ #当前高度的Rem_

            # if height >= pred_height and pred_step > 0: #预测下界并修改Rem_
                # under_bound=Rem[height-1]*Rem[height-1]//Rem[height-2] # 用N(h-1)^2/N(h-2)作为下界
                # if Rem_<under_bound:
                #     Rem_ += ((under_bound-Rem_)//Mod_) * Mod_
                # while Rem_<(Rem[height//2]*Rem[(height+1)//2]):
                #     Rem_+=Mod_

            # 当前高度未发生变化，之前的也未发生变化
            if Rem_ == Rem[height] and not changed:
                Rem_unchange[height] += 1 # 未变化次数+1
                now_height_0 += 1 

                # 未变化次数已经达到阈值，则已计算高度（不变次数达到要求的最大高度）+1
                if Rem_unchange[height] >= shred: 
                    now_height += 1

            # 当前高度发生变化，或之前已有高度发生变化
            else:
                Rem_unchange[height] = 0
                Rem[height] = Rem_ #更新该高度的当前结果
                changed = True
        
        Mod = Mod_ # 更新Mod

        # 不变的最小高度增加则输出（保存）已有结果
        if now_height > last_height:

            width_ = width_bias[0] + width_bias[1] * width  # 对宽进行偏置

            # 打印在控制台上，空格隔开（不换行）
            print("; ".join(f"{width_}x{height}: {Rem[height]}" for height in range(last_height + 1, now_height + 1)))

            # 保存在txt文件中，换行
            if save_rst:
                with open(f"enum_result\\{name}.txt", 'a') as rst_file:
                    rst_file.write("\n".join(f"{width_}x{height}: {Rem[height]}"
                                             for height in range(last_height + 1, now_height + 1)) + "\n")

        # 更新进度条
        now_height_0 -= (now_height - now_height_0) / (shred - 1)  # 预测恰好正在改变的高度位置
        show_progr(f"正在计算宽度width={width_bias[0]}+{width_bias[1]}x{width}，第{now_modNum}个素数{mod}的铺陈计数：",
                   (now_height_0, max_height + (now_height_0 - now_height)), index=2,
                   startTime=enum_startTime)

        last_height = now_height

        # 全都连续1次以上不变则停止该轮计数，素数已用尽仍未停止则报错
        if now_height == max_height:
            break
        elif now_modNum == len(mod_list):
            raise OverflowError
        # --------------------------------------------------

    print(
        f"# ↑已得到({width_bias[0]}+{width_bias[1]}x{width})xh, h≤{max_height}的铺陈总数，使用{now_modNum}个素数，用时{time.time() - enum_startTime}s")
    if save_rst:
        with open(f"enum_result\\{name}.txt", 'a') as rst_file:
            spentTime = time.time() - enum_startTime
            rst_file.write(
                f"# calculation time for width={width_bias[0]}+{width_bias[1]}x{width}: {spentTime}s, {now_modNum} primes used.\n\n")

def tiles_to_tensor(tiles_dict):
    """砖集转换成张量"""

    # 各类位置（8类）的各个方向的字符数
    # "x0y0" "y0" "x1y0"
    #     .________.       ._.
    #     |_|____|_|       |_|"xby0"
    # "x0"| | "" | |"x1"   | |"xb"
    charNum_dict = {pos: [max(t[0] for t in tiles_dict[pos]) + 1, max(t[1] for t in tiles_dict[pos]) + 1,
                          max(t[2] for t in tiles_dict[pos]) + 1, max(t[3] for t in tiles_dict[pos]) + 1]
                    for pos in tiles_dict}

    # 横向相接的边（共4对）的字符数取两者最大值（这种处理方法不能进一步提升效率）
    charNum = max(charNum_dict["x0y0"][3], charNum_dict["y0"][0])
    charNum_dict["x0y0"][3], charNum_dict["y0"][0] = charNum, charNum
    charNum = max(charNum_dict["y0"][3], charNum_dict["x1y0"][0])
    charNum_dict["y0"][3], charNum_dict["x1y0"][0] = charNum, charNum
    charNum = max(charNum_dict["x0"][3], charNum_dict[""][0])
    charNum_dict["x0"][3], charNum_dict[""][0] = charNum, charNum
    charNum = max(charNum_dict[""][3], charNum_dict["x1"][0])
    charNum_dict[""][3], charNum_dict["x1"][0] = charNum, charNum

    # 纵向相接的边（共4对）的字符数取两者最大值
    charNum = max(charNum_dict["x0y0"][2], charNum_dict["x0"][1])
    charNum_dict["x0y0"][2], charNum_dict["x0"][1] = charNum, charNum
    charNum = max(charNum_dict["y0"][2], charNum_dict[""][1])
    charNum_dict["y0"][2], charNum_dict[""][1] = charNum, charNum
    charNum = max(charNum_dict["x1y0"][2], charNum_dict["x1"][1])
    charNum_dict["x1y0"][2], charNum_dict["x1"][1] = charNum, charNum
    charNum = max(charNum_dict["xby0"][2], charNum_dict["xb"][1])
    charNum_dict["xby0"][2], charNum_dict["xb"][1] = charNum, charNum

    T_dict = {}
    for pos, tiles in tiles_dict.items():
        charNum = tuple(charNum_dict[pos])
        T = numpy.zeros((charNum[2], charNum[0] * charNum[1], charNum[3]), dtype=dtype)  # 根据各边字符数生成张量
        for t in tiles:
            p, a, b, q, k = t[0], t[1], t[2], t[3], (1 if len(t) == 4 else t[4])
            if p < charNum[0] and a < charNum[1] and b < charNum[2] and q < charNum[3]:
                T[b, p + a * charNum[0], q] = k
        T_dict[pos] = T

    print(charNum_dict)

    return T_dict, charNum_dict


def rect_til(width_range, height_range, tiles_dict, has_neg=None):
    """
    选定棋盘长宽范围和砖集，在矩形棋盘内铺陈并输出结果
    :param width_range: 宽度范围，格式为 (最小宽度，最大宽度)
    :param height_range: 高度范围，最大高度为关于宽度的一次函数h_max=w_max+a+b(w_max-w)，格式为 (a, b)，
    :param tiles_dict:
    :param has_neg:
    """
    global dtype, mod_list

    # 检查砖权重是否有负值，以决定数据类型
    if has_neg is None:
        has_neg = False
        for t in tiles_dict[""]:
            if len(t) == 5 and t[4] < 0:
                has_neg = True
        dtype = (numpy.int32 if has_neg else numpy.uint32) if dsize == 4 else (numpy.int16 if has_neg else numpy.uint16)

    # 右边缘位置砖集直接进行限制对一般位置砖集进行边界限制得到，左，上边缘位置若没有给出则这样得到。
    if "x0" not in tiles_dict:
        tiles_dict["x0"] = {t for t in tiles_dict[""] if t[0] == 0}
    tiles_dict["x1"] = {t for t in tiles_dict[""] if t[3] == 0}
    if "y0" not in tiles_dict:
        tiles_dict["y0"] = {t for t in tiles_dict[""] if t[1] == 0}
    if "x0y0" not in tiles_dict:
        tiles_dict["x0y0"] = {t for t in tiles_dict["x0"] if t[1] == 0}
    tiles_dict["x1y0"] = {t for t in tiles_dict["x1"] if t[1] == 0}
    if "xb" not in tiles_dict:
        tiles_dict["xb"] = {t for t in tiles_dict["x0"] if t[3] == 0}
    if "xby0" not in tiles_dict:
        tiles_dict["xby0"] = {t for t in tiles_dict["x0y0"] if t[3] == 0}

    T_dict, charNum_dict = tiles_to_tensor(tiles_dict)
    '''记录各类位置的砖的字典'''

    # 宽度范围是一个整数（下限），则根据内存限制和字符数给出上限，
    if len(width_range) == 1:
        theta, sigma = charNum_dict[""][0], charNum_dict[""][2]  # 一般位置横纵字符数
        print(f"theta, sigma={theta, sigma}")
        sigma_0, sigma_1 = charNum_dict["x0"][2], charNum_dict["x1"][2]  # 边缘位置纵向字符数
        print(f"sigma_0, sigma_1={sigma_0, sigma_1}")
        max_width = int(numpy.log2(mem_size / dsize /2 / sigma_0 / sigma_1 / theta) / numpy.log2(sigma)) + 2
        print(f"已根据字符数(θ, σ, σ_0, σ_1)={(theta,sigma,sigma_0,sigma_1)}"
              f"给出最大宽度width={width_bias[0]}+{width_bias[1]}x{max_width}={width_bias[0]+width_bias[1]*max_width}, "
              f"内存占用{dsize*theta*sigma_0*sigma_1*(sigma**(max_width-2))/(1024**3)}GB<{mem_size/(1024**3)}GB")
        width_range = (width_range[0], max_width)

    width_range_ = (width_bias[0] + width_range[0] * width_bias[1], width_bias[0] + width_range[1] * width_bias[1])

    # 打印当前参数
    print(f"# 任务名name={name}")
    print(f"# 一般位置砖集tiles={tiles_dict[""]}")
    print(f"# 左边缘位置砖集tiles_x0={tiles_dict["x0"]}")
    print(f"# 数据类型dtype={dtype}, 内存限制={mem_size // (1024 ** 3)}GB, 是否保存结果save_rst={save_rst}")
    print(f"# 宽度width={width_bias[0]}+{width_bias[1]}x({width_range[0]}..{width_range[1]}), " +
          f"高度height={1}..{height_range[0]+width_range_[1]} + {height_range[1]}*({width_range_[1]}-width)")
    if save_rst:
        with open(f"enum_result\\{name}.txt", 'a') as rst_file:
            # 英文开头
            rst_file.write(f"# result name: {name}\n")
            rst_file.write(f"# tile set at (1<i<l, 1<j): T={tiles_dict[""]}\n")
            rst_file.write(f"# tile set at (1, 1<j): T_x0={tiles_dict[""]}\n")
            rst_file.write(f"# width={width_bias[0]}+{width_bias[1]}x({width_range[0]}..{width_range[1]})")
            rst_file.write(
                f"# height={1}..{height_range[0]+width_range_[1]} + {height_range[1]}*({width_range[1]}-width)\n")
            rst_file.write("# results:\n")

    # 生成素数表
    renew_prime_list(T_dict[""], T_dict["x0"])

    sum_startTime = time.time()

    for width in range(width_range[0], width_range[1] + 1):

        ini_W = numpy.array([1], dtype=dtype)  # 初始状态张量

        # 待铺陈的张量列表
        max_height = (height_range[0] + width_range_[1]) + height_range[1] * (width_range_[1] - width * width_bias[1])
        if width == 1:
            T_list = [T_dict["xby0"]] + [T_dict["xb"] for _ in range(max_height - 1)]
        else:
            T_list = [T_dict["x0y0"]] + [T_dict["y0"] for _ in range(width - 2)] + [T_dict["x1y0"]]  # 上边缘
            T_list += ([T_dict["x0"]] + [T_dict[""] for _ in range(width - 2)] + [T_dict["x1"]]) * (max_height - 1)

        # 进行铺陈计数
        til_enum(ini_W, T_list, width=width)

    print(f"已完成，总用时{time.time() - sum_startTime}s")
    if save_rst:
        with open(f"enum_result\\{name}.txt", 'a') as rst_file:
            rst_file.write(f"# total calculation time={time.time() - sum_startTime}s\n# \n")


"""↓↓↓参数设置："""

progr_delay = 10
'''显示进度的最短延迟秒数'''

dsize = 2
'''单位数据的长度，单位是字节'''
# mem_size = 3 * (1024 ** 3)  # 3GB，每个张量1.5GB
dtype = numpy.uint32 if dsize == 4 else numpy.uint16
'''数据类型，仅限4或2字节的类型'''
# mem_size = 6 * (1024 ** 3)  # 6GB
mem_size = 9 * 1024 ** 3
'''分配的内存大小，单位是字节'''
save_rst = False
'''是否保存结果'''
modNum = 100
'''选取的素数个数'''
width_bias = (0, 1)
'''输出的结果的宽的偏置，为width_=bias[0]+bias[1]*width'''
# pred_height = 4
# '''进行下界预判的最小高度'''
# pred_step = 1
# '''进行下界预判的步长'''

"""↓↓↓主循环"""

name = "I1+O4_40"
'''当前任务名称'''
tiles = set()
'''一般位置的砖集'''
chars_y = {'####': 0, '##RR': 1, '#LL#': 2, 'RR##': 3, 'RRRR': 4, 'L###': 5, 'L#RR': 6, 'LLL#': 7}
'''纵向字母表及其编码'''
for A, a in chars_y.items():
    for B, b in chars_y.items():
        f = lambda x, y: (x == '#' or y == '#')  # 没有冲突
        if f(A[0], B[0]) and f(A[1], B[1]) and f(A[2], B[2]) and f(A[3], B[3]):  # 各个位置都没有冲突
            p = 1 if A[0] == 'L' or B[0] == 'L' else 0  # 左侧字符取决于上下侧字符首位
            tiles.add((p, a, b, 0))  # 添加右侧字符为0的砖

            # 最右端有空位，则再添加右侧字符为1的砖
            if A[3] == '#' and B[3] == '#':
                tiles.add((p, a, b, 1))

save_rst = True # 将保存结果设置为True（默认为False）

# 4w+2情形
width_bias = (-2, 4)  # 宽偏置
OO, RR=0, 1
tiles_x0={(0,OO,OO,0), (0,OO,OO,1), (0,RR,OO,0), (0,OO,RR,0)} # 此时左边缘砖集是两个砖的横向拼接
tiles_dict = {"": tiles, "x0": tiles_x0}
rect_til((1, 10), (3, 1), tiles_dict)

# 4w+3情形
width_bias = (-1, 4)  # 宽偏置
OOO, ORR, LLO= 0, 1, 2
tiles_x0={(0,OOO,OOO,0), (0,OOO,OOO,1), (0,ORR,OOO,0), (0,OOO,ORR,0),(0,LLO,OOO,0),(0,LLO,OOO,1),(0,OOO,LLO,0),(0,OOO,LLO,1)} # 此时左边缘砖集是3个砖的横向拼接
tiles_dict = {"": tiles, "x0": tiles_x0}
rect_til((1, 10), (2, 1), tiles_dict)

# 4w情形
width_bias = (0, 4)  # 宽偏置
tiles_dict = {"": tiles}
rect_til((1, 9), (5, 1), tiles_dict)

# 4w+1情形
width_bias = (1, 4)  # 宽偏置
tiles_x0={(0,t[1],t[2],t[3]) for t in tiles} # 此时的左边缘砖集等同于一般位置砖集不再限制左侧字符
tiles_dict = {"": tiles, "x0": tiles_x0}
rect_til((1, 9), (4, 1), tiles_dict)
