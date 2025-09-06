import React, { useState, useCallback, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Play, Download, Thermometer, Timer, Settings } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { steadyStateAPI, CoolingSimResponse } from '@/lib/api';
import LoadingSpinner, { ButtonLoader } from '../components/LoadingSpinner';
import ErrorAlert, { SuccessAlert } from '../components/ErrorAlert';
import FormInput from '../components/FormInput';
import BatchProcessor from '../components/BatchProcessor';

interface FormData {
  T1: number;  // 温度T1值
  T2: number;  // 温度T2值
}

interface ChartData {
  time: number;
  T1: number;
  T2: number;
  dTdt: number;
  deltaRatio: number;
}

export default function Cooling() {
  const [formData, setFormData] = useState<FormData>({
    T1: 99,   // 温度T1值
    T2: 50,   // 温度T2值
  });

  const [loading, setLoading] = useState(false);
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [results, setResults] = useState<CoolingSimResponse | null>(null);

  const [showBatch, setShowBatch] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleInputChange = (field: keyof FormData, value: number) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const [analysisResult, setAnalysisResult] = useState<string>('');

  const handleSimulate = useCallback(async (retryCount = 0) => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);
      
      // 调用稳态法ΔT Δt分析
      const response = await fetch('/api/steady/steady-state-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          T1: formData.T1,
          T2: formData.T2,
        }),
      });
      
      if (!response.ok) {
        throw new Error('分析请求失败');
      }
      
      const result = await response.json();
      setAnalysisResult(result.analysis_result);
      setSuccess('稳态法ΔT Δt分析完成！');
    } catch (err: any) {
      const errorMessage = err.message || '分析失败';
      
      // 网络错误且重试次数小于3次时自动重试
      if (!err.response && retryCount < 3) {
        setTimeout(() => {
          handleSimulate(retryCount + 1);
        }, 1000 * (retryCount + 1));
        setError(`网络连接失败，正在重试... (${retryCount + 1}/3)`);
      } else {
        setError(errorMessage);
      }
    } finally {
      if (retryCount === 0) {
        setLoading(false);
      }
    }
  }, [formData]);

  const exportData = () => {
    if (!analysisResult) return;
    
    const blob = new Blob([analysisResult], { type: 'text/plain;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `steady_state_analysis_${new Date().toISOString().slice(0, 10)}.txt`;
    link.click();
  };





  return (
    <div className="min-h-screen bg-gray-50">
      {/* 头部导航 */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link to="/" className="flex items-center gap-2 text-gray-600 hover:text-gray-900">
                <ArrowLeft className="w-5 h-5" />
                返回
              </Link>
              <h1 className="text-2xl font-bold text-gray-900">稳态法ΔT Δt分析实验台</h1>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={exportData}
                disabled={!analysisResult}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                <Download className="w-4 h-4" />
                导出分析结果
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* 左侧参数面板 */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">仿真参数</h2>
              
              {/* 基本参数 */}
              <div className="space-y-4">
                <FormInput
                  label="温度T1值"
                  value={formData.T1}
                  onChange={(value) => handleInputChange('T1', value as number)}
                  type="number"
                  unit="°C"
                  min={38.2}
                  max={100.0}
                  step={0.1}
                  hint="范围: 38.2-100.0°C，分析目标温度点1"
                  required
                />
                
                <FormInput
                  label="温度T2值"
                  value={formData.T2}
                  onChange={(value) => handleInputChange('T2', value as number)}
                  type="number"
                  unit="°C"
                  min={38.2}
                  max={100.0}
                  step={0.1}
                  hint="范围: 38.2-100.0°C，分析目标温度点2"
                  required
                />
              </div>



              {/* 操作按钮 */}
              <div className="mt-6 space-y-3">
                <div className="flex gap-2">
                  <button
                    onClick={() => setShowBatch(false)}
                    className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                      !showBatch 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    单次实验
                  </button>
                  <button
                    onClick={() => setShowBatch(true)}
                    className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                      showBatch 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    批量处理
                  </button>
                </div>
                
                {!showBatch && (
                  <button
                    onClick={() => handleSimulate()}
                    disabled={loading}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
                  >
                    {loading ? (
                      <ButtonLoader text="仿真中..." />
                    ) : (
                      <>
                      <Play className="w-4 h-4" />
                      开始稳态法ΔT Δt分析
                    </>
                    )}
                  </button>
                )}
              </div>

              {/* 状态提示 */}
              <ErrorAlert 
                error={error} 
                onDismiss={() => setError(null)}
                onRetry={handleSimulate}
                className="mt-4"
              />
              
              {success && (
                <SuccessAlert 
                  message={success}
                  onDismiss={() => setSuccess(null)}
                  className="mt-4"
                />
              )}
            </div>


          </div>

          {/* 右侧图表 */}
          <div className="lg:col-span-2 space-y-6">
            {/* 批量处理组件 */}
            {showBatch && (
              <BatchProcessor
                method="cooling"
                material="copper"
                onResultsChange={(results) => {
                  if (results) {
                    console.log('批量处理完成:', results);
                    setSuccess(`批量处理完成！成功处理 ${results.success_count}/${results.total_rows} 个样本`);
                  }
                }}
              />
            )}
            
            {/* 稳态法ΔT Δt分析结果 */}
            {!showBatch && analysisResult && (
              <div className="bg-white rounded-lg shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">稳态法ΔT Δt分析结果</h3>
                <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm overflow-auto max-h-96">
                  <div className="whitespace-pre-wrap">
                    {analysisResult}
                  </div>
                </div>
              </div>
            )}

            {/* 实验说明 */}
            {!showBatch && (
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">实验说明</h3>
              <div className="prose text-sm text-gray-600">
                <p className="mb-3">
                  <strong>稳态法ΔT Δt分析实验</strong>基于铜材料的热传导特性，通过分析表面温度T1和内部温度T2之间的关系，计算温度差时间比等关键参数。
                </p>
                <p className="mb-3">
                  <strong>物理原理：</strong>根据牛顿冷却定律和热传导理论，结合对流散热和辐射散热，建立铜材料的冷却模型。
                </p>
                <p className="mb-3">
                  <strong>应用场景：</strong>材料热物性分析、导热系数测定、热传导过程研究、工业热处理优化等。
                </p>
                <p>
                  <strong>参数说明：</strong>
                </p>
                <ul className="list-disc list-inside mt-2 space-y-1">
                  <li>温度T1值: 分析目标温度点1，用于计算特征比值</li>
                  <li>温度T2值: 分析目标温度点2，用于计算特征比值</li>
                  <li>特征比值: ΔT/Δt，表示温度差与时间差的比值</li>
                  <li>时间差: T1和T2温度点对应的时间差</li>
                  <li>温度差: T1和T2之间的温度差值</li>
                  <li>分析输出: 包含详细的热分析数据和可视化结果</li>
                </ul>
                <p className="mt-3">
                  <strong>实验流程：</strong>输入T1和T2温度值 → 生成铜材料冷却曲线 → 计算温度变化率和内部温度 → 分析温度差时间比 → 输出综合分析结果
                </p>
              </div>
            </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}