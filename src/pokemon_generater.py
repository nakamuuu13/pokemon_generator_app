from importer import *

#Pixel normalizationをする関数
#ピクセルごとに特徴量を正規化する
#チャネル方向に2乗平均をとってsqrtをした値で割り算をする
#generatorのみに使われる
class PixelNorm(nn.Module):
	def forward(self, x):
		#0除算を防ぐために十分小さい数epsを用意する
		eps = 1e-7
		#チャネル方向(dim=1)に2乗(x**2)平均をとる
		mean = torch.mean(x**2,dim=1,keepdims=True)
		return x / (torch.sqrt(mean)+eps)

#学習率の平滑化をする関数
#各レイヤの重みを入力チャネルサイズで正規化する。Heの初期化と似た効果を期待するもの。
class EqualizeLearningRate(nn.Module):
	def forward(self, x, gain=2):
		scale = (gain/x.shape[1])**0.5
		return x * scale

#畳み込み層のモジュール「Conv2d」
#処理を1まとめにして扱いやすくしておく
#層の途中にあるReflectionPad2dはzero paddingと似た役割をするが
#zero paddingと比べて元の入力に近い分布を実現できるため
#生成された画像の端付近にアーティファクトができにくくなる
class Conv2d(nn.Module):
	'''
	引数:
		inch: (int)  入力チャネル数
		outch: (int) 出力チャネル数
		kernel_size: (int) カーネルの大きさ
		padding: (int) パディング
	'''
	def __init__(self, inch, outch, kernel_size, padding=0):
		super().__init__()
		self.layers = nn.Sequential(
			EqualizeLearningRate(),
			nn.ReflectionPad2d(padding),
			nn.Conv2d(inch, outch, kernel_size, padding=0),
			PixelNorm(),
		)
		nn.init.kaiming_normal_(self.layers[2].weight)
	def forward(self, x):
		return self.layers(x)

#generator用畳み込み層
#generatorの最初の層のみUpsampleなし
class ConvModuleG(nn.Module):
	def __init__(self, out_size, inch, outch, first=False):
		super().__init__()
		if first:
			layers = [
				Conv2d(inch, outch, 3, padding=1),
				nn.LeakyReLU(0.2, inplace=False),
				Conv2d(outch, outch, 3, padding=1),
				nn.LeakyReLU(0.2, inplace=False),
			]
		else:
			layers = [
				nn.Upsample((out_size, out_size), mode='nearest'),
				Conv2d(inch, outch, 3, padding=1),
				nn.LeakyReLU(0.2, inplace=False),
				Conv2d(outch, outch, 3, padding=1),
				nn.LeakyReLU(0.2, inplace=False),
			]
		self.layers = nn.Sequential(*layers)
	def forward(self, x):
		return self.layers(x)


class Generator(nn.Module):
	def __init__(self):
		super().__init__()
		#畳み込みモジュールの設定を1つずつしていく
		inchs  = np.array([512,256,128,64,32, 16,  8], dtype=np.uint32)
		outchs = np.array([256,128, 64,32,16,  8,  4], dtype=np.uint32)
		sizes  = np.array([  4,  8, 16,32,64,128,256], dtype=np.uint32)
		#最初の層のみ、それを示すフラグをTrueにしておく
		firsts = np.array([True,False,False,False,False,False,False], dtype=np.bool_)
		#blockには畳み込み層を格納、toRGBsは入力されたデータを出力画像(RGB3チャネル)に変換するための層を格納
		blocks, toRGBs = [], []
		for s, inch, outch, first in zip(sizes, inchs, outchs, firsts):
			blocks.append(ConvModuleG(s, inch, outch, first))
			toRGBs.append(nn.Conv2d(outch, 3, 1, padding=0))
		self.blocks = nn.ModuleList(blocks)
		self.toRGBs = nn.ModuleList(toRGBs)
	def forward(self, x, res, eps=1e-7):
		# to image
		n,c = x.shape
		x = x.reshape(n,c//16,4,4)
		#何層目まで畳み込みを計算するかをresとする
		res = min(res, len(self.blocks))#resが畳み込み層の数より大きくならないようにする
		#0~(nlayer-1)層目まで畳み込みを計算する
		nlayer = max(int(res-eps), 0)
		for i in range(nlayer):
			x = self.blocks[i](x)
		#最後の層（nlayer番目）
		x_last = self.blocks[nlayer](x)
		dst_big = self.toRGBs[nlayer](x_last)
		if nlayer==0:
			x = dst_big
		else:
			#1個下の解像度と混ぜ合わせるようにしながら学習を行う
			x_sml = F.interpolate(x, x_last.shape[2:4], mode='nearest')
			dst_sml = self.toRGBs[nlayer-1](x_sml)
			alpha = res - int(res-eps)
			x = (1-alpha)*dst_sml + alpha*dst_big
		#return x, n, res
		return torch.sigmoid(x)