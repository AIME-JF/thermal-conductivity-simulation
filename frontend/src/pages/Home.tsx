import { Link } from 'react-router-dom';
import { Thermometer, Zap, Settings, Snowflake } from 'lucide-react';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* 头部 */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">热传导实验虚拟仿真平台</h1>
              <p className="text-gray-600 mt-2">基于神经网络修正的导热系数测量实验</p>
            </div>
            <Link
              to="/admin"
              className="flex items-center gap-2 px-4 py-2 text-gray-600 hover:text-gray-900 transition-colors"
            >
              <Settings className="w-5 h-5" />
              管理员
            </Link>
          </div>
        </div>
      </header>

      {/* 主要内容 */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-12">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">选择实验方法</h2>
          <p className="text-gray-600 max-w-2xl mx-auto">
            本平台提供两种导热系数测量方法：稳态法适用于橡胶等材料，准稳态法适用于有机玻璃和橡胶材料。
            每种方法都集成了神经网络修正算法，提供更准确的测量结果。
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {/* 稳态法卡片 */}
          <Link to="/steady" className="group">
            <div className="bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 transform group-hover:-translate-y-1 p-8">
              <div className="flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-6 mx-auto group-hover:bg-blue-200 transition-colors">
                <Thermometer className="w-8 h-8 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4 text-center">稳态法</h3>
              <div className="space-y-3 text-sm text-gray-600">
                <div className="flex items-start gap-2">
                  <span className="font-medium text-gray-900">适用材料：</span>
                  <span>橡胶等低导热材料</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="font-medium text-gray-900">主要观测量：</span>
                  <span>热端温度T1、冷端温度T2</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="font-medium text-gray-900">核心算法：</span>
                  <span>T2线性修正 + 神经网络预测</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="font-medium text-gray-900">特色功能：</span>
                  <span>独立冷却仿真实验、实时温度曲线</span>
                </div>
              </div>
              <div className="mt-6 text-center">
                <span className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg group-hover:bg-blue-700 transition-colors">
                  开始实验
                </span>
              </div>
            </div>
          </Link>

          {/* 准稳态法卡片 */}
          <Link to="/quasi" className="group">
            <div className="bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 transform group-hover:-translate-y-1 p-8">
              <div className="flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-6 mx-auto group-hover:bg-green-200 transition-colors">
                <Zap className="w-8 h-8 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4 text-center">准稳态法</h3>
              <div className="space-y-3 text-sm text-gray-600">
                <div className="flex items-start gap-2">
                  <span className="font-medium text-gray-900">适用材料：</span>
                  <span>有机玻璃、橡胶</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="font-medium text-gray-900">主要观测量：</span>
                  <span>电压V_t、电压变化率ΔV</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="font-medium text-gray-900">核心算法：</span>
                  <span>多任务神经网络双输出预测</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="font-medium text-gray-900">特色功能：</span>
                  <span>同时预测导热系数λ和比热容c</span>
                </div>
              </div>
              <div className="mt-6 text-center">
                <span className="inline-flex items-center px-4 py-2 bg-green-600 text-white rounded-lg group-hover:bg-green-700 transition-colors">
                  开始实验
                </span>
              </div>
            </div>
          </Link>

          {/* 冷却仿真卡片 */}
          <Link to="/cooling" className="group">
            <div className="bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 transform group-hover:-translate-y-1 p-8">
              <div className="flex items-center justify-center w-16 h-16 bg-purple-100 rounded-full mb-6 mx-auto group-hover:bg-purple-200 transition-colors">
                <Snowflake className="w-8 h-8 text-purple-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4 text-center">冷却仿真</h3>
              <div className="space-y-3 text-sm text-gray-600">
                <div className="flex items-start gap-2">
                  <span className="font-medium text-gray-900">适用材料：</span>
                  <span>铜块等金属材料</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="font-medium text-gray-900">主要观测量：</span>
                  <span>温度变化曲线、冷却速率</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="font-medium text-gray-900">核心算法：</span>
                  <span>牛顿冷却定律仿真</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="font-medium text-gray-900">特色功能：</span>
                  <span>动态冷却过程可视化</span>
                </div>
              </div>
              <div className="mt-6 text-center">
                <span className="inline-flex items-center px-4 py-2 bg-purple-600 text-white rounded-lg group-hover:bg-purple-700 transition-colors">
                  开始实验
                </span>
              </div>
            </div>
          </Link>
        </div>

        {/* 底部信息 */}
        <div className="mt-16 text-center">
          <div className="bg-white rounded-lg shadow-sm p-6 max-w-3xl mx-auto">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">实验平台特色</h3>
            <div className="grid md:grid-cols-3 gap-6 text-sm">
              <div className="text-center">
                <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-3">
                  <span className="text-purple-600 font-bold">AI</span>
                </div>
                <h4 className="font-medium text-gray-900 mb-2">神经网络修正</h4>
                <p className="text-gray-600">基于深度学习的误差修正算法，提高测量精度</p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-3">
                  <span className="text-orange-600 font-bold">📊</span>
                </div>
                <h4 className="font-medium text-gray-900 mb-2">实时可视化</h4>
                <p className="text-gray-600">动态温度曲线、数据分析图表</p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-cyan-100 rounded-full flex items-center justify-center mx-auto mb-3">
                  <span className="text-cyan-600 font-bold">📁</span>
                </div>
                <h4 className="font-medium text-gray-900 mb-2">数据管理</h4>
                <p className="text-gray-600">支持CSV导入导出、PDF报告生成</p>
              </div>
            </div>
          </div>
        </div>

        {/* 数据来源说明 */}
        <div className="mt-16 pt-8 border-t border-gray-200">
          <div className="bg-gray-50 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 text-center">数据来源说明</h3>
            <div className="text-sm text-gray-600 space-y-3">
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                <div>
                  <span className="font-medium text-gray-900">实验数据：</span>
                  本平台的实验数据来源于南京航空航天大学物理实验室的真实测量数据，包括稳态法和准稳态法的导热系数测量实验。
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                <div>
                  <span className="font-medium text-gray-900">神经网络模型：</span>
                  误差修正算法基于深度学习神经网络训练，使用大量实验数据进行模型优化，提供更准确的导热系数预测结果。
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-purple-500 rounded-full mt-2 flex-shrink-0"></div>
                <div>
                  <span className="font-medium text-gray-900">技术支持：</span>
                  平台采用现代Web技术栈开发，确保实验数据的准确性和用户体验的流畅性。
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}