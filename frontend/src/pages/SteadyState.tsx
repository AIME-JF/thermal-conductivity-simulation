import React, { useState, useCallback, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Play, Upload, Download, FileText, Settings, Calculator } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { steadyStateAPI, SteadyStateRequest, SteadyStateResponse, CoolingSimResponse } from '@/lib/api';
import LoadingSpinner, { ButtonLoader } from '../components/LoadingSpinner';
import ErrorAlert, { SuccessAlert } from '../components/ErrorAlert';
import FormInput, { ModeSelector, MaterialSelector } from '../components/FormInput';
import BatchProcessor from '../components/BatchProcessor';
import ModelSelector from '../components/ModelSelector';

interface FormData {
  T1: number;
  T2: number;
  mode: 'auto' | 'manual';
  duration: number;
  noise: number;
  material: 'copper';
  selectedModel: string; // 添加模型选择字段
  // 材料参数（可编辑）
  thermalConductivity: number;
  density: number;
  specificHeat: number;
  thickness: number;
}

interface ChartData {
  time: number;
  T1: number;
  T2: number;
  dTdt: number;
  deltaRatio: number;
}

// 材料默认参数
const MATERIAL_DEFAULTS = {
  copper: {
    thermalConductivity: 401,
    density: 8960,
    specificHeat: 385,
    thickness: 0.01,
  },
};

export default function SteadyState() {
  const [formData, setFormData] = useState<FormData>({
    T1: 75,
    T2: 52,
    mode: 'auto',
    duration: 3600,
    noise: 0.1,
    material: 'copper',
    selectedModel: 'default', // 默认使用系统模型
    ...MATERIAL_DEFAULTS.copper,
  });

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SteadyStateResponse | null>(null);
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showBatch, setShowBatch] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});

  const validateInput = (field: keyof FormData, value: number): string | null => {
    switch (field) {
      case 'T1':
        if (value <= 0 || value > 1000) return 'T1温度必须在0-1000°C范围内';
        break;
      case 'T2':
        if (value <= 0 || value > 1000) return 'T2温度必须在0-1000°C范围内';
        break;
      case 'duration':
        if (value < 300 || value > 7200) return '仿真时长必须在300-7200秒范围内';
        break;
      case 'noise':
        if (value < 0 || value > 1) return '噪声水平必须在0-1范围内';
        break;
      case 'thermalConductivity':
        if (value <= 0 || value > 500) return '导热系数必须在0-500 W/(m·K)范围内';
        break;
      case 'density':
        if (value <= 0 || value > 20000) return '密度必须在0-20000 kg/m³范围内';
        break;
      case 'specificHeat':
        if (value <= 0 || value > 5000) return '比热容必须在0-5000 J/(kg·K)范围内';
        break;
      case 'thickness':
        if (value <= 0 || value > 0.1) return '厚度必须在0-0.1 m范围内';
        break;
    }
    return null;
  };

  const handleInputChange = (field: keyof FormData, value: number | string) => {
    // 清除之前的验证错误
    setValidationErrors(prev => {
      const newErrors = { ...prev };
      delete newErrors[field];
      return newErrors;
    });
    
    // 对于字符串字段（如selectedModel、mode），直接使用原值
    if (field === 'selectedModel' || field === 'mode' || field === 'material') {
      setFormData(prev => ({ ...prev, [field]: value }));
      return;
    }
    
    // 对于数字字段，进行转换和验证
    const numValue = typeof value === 'string' ? parseFloat(value) : value;
    
    // 验证输入
    if (!isNaN(numValue)) {
      const validationError = validateInput(field, numValue);
      if (validationError) {
        setValidationErrors(prev => ({ ...prev, [field]: validationError }));
      }
    }
    
    setFormData(prev => ({ ...prev, [field]: numValue }));
  };

  const handleMaterialChange = (material: 'copper') => {
    setFormData(prev => ({
      ...prev,
      material,
      ...MATERIAL_DEFAULTS[material],
    }));
  };

  const handleSimulateCooling = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const coolingData = await steadyStateAPI.simulateCooling({
        duration: formData.duration,
        noise: formData.noise,
      });
      
      // 转换数据格式用于图表显示（优化性能）
      const chartPoints: ChartData[] = coolingData.time.map((time, index) => ({
        time: Math.round(time),
        T1: Number(coolingData.T1[index].toFixed(2)),
        T2: Number(coolingData.T2[index].toFixed(2)),
        dTdt: Number(coolingData.dTdt[index].toFixed(4)),
        deltaRatio: Number(coolingData.deltaRatios[index].toFixed(4)),
      }));
      
      // 减少数据点以提高渲染性能
      const optimizedChartPoints = chartPoints.filter((_, index) => 
        index % Math.max(1, Math.floor(chartPoints.length / 200)) === 0
      );
      
      setChartData(optimizedChartPoints);
      
      // 自动设置T1和T3为仿真数据的初始值
      if (chartPoints.length > 0) {
        setFormData(prev => ({
          ...prev,
          T1: chartPoints[0].T1,
          T2: chartPoints[0].T2, // 使用T2作为T2的初始值
        }));
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || '冷却仿真失败');
    } finally {
      setLoading(false);
    }
  }, [formData.duration, formData.noise]);

  const handlePredict = useCallback(async (retryCount = 0) => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);
      
      // 客户端验证
      if (formData.T1 <= formData.T2) {
        setError('T1温度必须大于T2温度');
        setLoading(false);
        return;
      }
      
      if (Math.abs(formData.T1 - formData.T2) < 5) {
        setError('T1和T2温差必须大于5°C');
        setLoading(false);
        return;
      }
      
      // 检查是否有验证错误
      if (Object.keys(validationErrors).length > 0) {
        setError('请修正输入参数错误后再试');
        setLoading(false);
        return;
      }
      
      const request: SteadyStateRequest = {
        T1: formData.T1,
        T2: formData.T2,
        selectedModel: formData.selectedModel,
        options: {
          use_auto_cooling: formData.mode === 'auto',
        },
      };
      
      const result = await steadyStateAPI.predict(request);
      setResults(result);
      setSuccess('预测完成！结果已更新。');
    } catch (err: any) {
      const errorMessage = err.message || err.response?.data?.detail || '预测失败';
      
      // 网络错误且重试次数小于3次时自动重试
      if (!err.response && retryCount < 3) {
        setTimeout(() => {
          handlePredict(retryCount + 1);
        }, 1000 * (retryCount + 1)); // 递增延迟重试
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
    if (!results || chartData.length === 0) return;
    
    const csvContent = [
      ['时间(s)', 'T1(°C)', 'T2(°C)', 'dT/dt', 'ΔT/Δt'],
      ...chartData.map(point => [
        point.time,
        point.T1,
        point.T2,
        point.dTdt,
        point.deltaRatio,
      ]),
    ].map(row => row.join(',')).join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `steady_state_results_${new Date().toISOString().slice(0, 10)}.csv`;
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
              <h1 className="text-2xl font-bold text-gray-900">稳态法实验台</h1>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={exportData}
                disabled={!results || chartData.length === 0}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                <Download className="w-4 h-4" />
                导出数据
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
              <h2 className="text-lg font-semibold text-gray-900 mb-4">实验参数</h2>
              
              {/* 模型选择 */}
              <ModelSelector
                value={formData.selectedModel}
                onChange={(value) => handleInputChange('selectedModel', value)}
                modelType="steady"
                className="mb-6"
              />
              
              {/* 材料选择 */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  实验材料
                </label>
                <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                    <span className="font-medium text-blue-900">金属铜</span>
                    <span className="text-sm text-blue-600">高导热金属材料</span>
                  </div>
                </div>
              </div>
              
              {/* 实验模式选择 */}
              <ModeSelector
                value={formData.mode}
                onChange={(value) => handleInputChange('mode', value)}
                options={[
                  { value: 'auto', label: '自动仿真', icon: <Calculator className="w-4 h-4" /> },
                  { value: 'manual', label: '手动输入', icon: <Settings className="w-4 h-4" /> }
                ]}
                className="mb-6"
              />

              {/* 温度参数 */}
              <div className="space-y-4">
                <FormInput
                  label="热端温度 T1"
                  value={formData.T1}
                  onChange={(value) => handleInputChange('T1', value as number)}
                  type="number"
                  unit="°C"
                  min={25}
                  max={150}
                  step={0.1}
                  hint="范围: 25-150°C"
                  error={validationErrors.T1}
                  required
                />
                
                <FormInput
                  label="冷端温度 T2"
                  value={formData.T2}
                  onChange={(value) => handleInputChange('T2', value as number)}
                  type="number"
                  unit="°C"
                  min={20}
                  max={100}
                  step={0.1}
                  hint="范围: 20-100°C"
                  error={validationErrors.T2}
                  required
                />
              </div>

              {/* 高级设置 */}
              <div className="mt-6">
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900"
                >
                  <Settings className="w-4 h-4" />
                  高级设置
                </button>
                
                {showAdvanced && (
                  <div className="mt-4 space-y-4 p-4 bg-gray-50 rounded-lg">
                    <FormInput
                      label="仿真时长"
                      value={formData.duration}
                      onChange={(value) => handleInputChange('duration', value as number)}
                      type="number"
                      unit="秒"
                      min={300}
                      max={7200}
                      step={60}
                      hint="范围: 300-7200秒 (5分钟-2小时)"
                      required
                    />
                    
                    <FormInput
                      label="噪声水平"
                      value={formData.noise}
                      onChange={(value) => handleInputChange('noise', value as number)}
                      type="number"
                      min={0}
                      max={1}
                      step={0.01}
                      hint="范围: 0-1，模拟实际测量误差"
                      required
                    />
                    
                    <div className="border-t pt-4 mt-4">
                      <h4 className="text-sm font-medium text-gray-700 mb-3">材料参数</h4>
                      <div className="grid grid-cols-2 gap-3">
                        <FormInput
                          label="导热系数"
                          value={formData.thermalConductivity}
                          onChange={(value) => handleInputChange('thermalConductivity', value as number)}
                          type="number"
                          unit="W/(m·K)"
                          min={0.1}
                          max={500}
                          step={0.1}
                          required
                        />
                        
                        <FormInput
                          label="密度"
                          value={formData.density}
                          onChange={(value) => handleInputChange('density', value as number)}
                          type="number"
                          unit="kg/m³"
                          min={100}
                          max={20000}
                          step={10}
                          required
                        />
                        
                        <FormInput
                          label="比热容"
                          value={formData.specificHeat}
                          onChange={(value) => handleInputChange('specificHeat', value as number)}
                          type="number"
                          unit="J/(kg·K)"
                          min={100}
                          max={2000}
                          step={10}
                          required
                        />
                        
                        <FormInput
                          label="厚度"
                          value={formData.thickness}
                          onChange={(value) => handleInputChange('thickness', value as number)}
                          type="number"
                          unit="m"
                          min={0.001}
                          max={0.1}
                          step={0.001}
                          required
                        />
                      </div>
                    </div>
                  </div>
                )}
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
                  <div className="space-y-3">
                    {formData.mode === 'auto' && (
                      <button
                        onClick={handleSimulateCooling}
                        disabled={loading}
                        className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
                      >
                        {loading ? (
                          <ButtonLoader text="仿真中..." />
                        ) : (
                          <>
                            <Play className="w-4 h-4" />
                            开始冷却仿真
                          </>
                        )}
                      </button>
                    )}
                    
                    <button
                      onClick={() => handlePredict()}
                      disabled={loading || (formData.mode === 'auto' && chartData.length === 0)}
                      className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400"
                    >
                      {loading ? (
                        <ButtonLoader text="计算中..." />
                      ) : (
                        <>
                          <Calculator className="w-4 h-4" />
                          计算导热系数
                        </>
                      )}
                    </button>
                  </div>
                )}
              </div>

              {/* 状态提示 */}
              <ErrorAlert 
                error={error} 
                onDismiss={() => setError(null)}
                onRetry={handlePredict}
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

          {/* 右侧图表和结果 */}
          <div className="lg:col-span-2 space-y-6">
            {/* 批量处理组件 */}
             {showBatch && (
               <BatchProcessor
                 method="steady"
                 material={formData.material}
                 onResultsChange={(results) => {
                   if (results) {
                     console.log('批量处理完成:', results);
                     setSuccess(`批量处理完成！成功处理 ${results.success_count}/${results.total_rows} 个样本`);
                   }
                 }}
               />
             )}
            {/* 温度曲线图 */}
            {chartData.length > 0 && (
              <div className="bg-white rounded-lg shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">温度变化曲线</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="time" 
                        label={{ value: '时间 (s)', position: 'insideBottom', offset: -5 }}
                      />
                      <YAxis 
                        label={{ value: '温度 (°C)', angle: -90, position: 'insideLeft' }}
                      />
                      <Tooltip 
                        formatter={(value: number, name: string) => [
                          `${value.toFixed(2)}${name.includes('T') ? '°C' : ''}`,
                          name
                        ]}
                      />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="T1" 
                        stroke="#ef4444" 
                        strokeWidth={2}
                        name="热端温度 T1"
                        dot={false}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="T2" 
                        stroke="#3b82f6" 
                        strokeWidth={2}
                        name="冷端温度 T2"
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* ΔT/Δt 曲线图 */}
            {chartData.length > 0 && (
              <div className="bg-white rounded-lg shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">温差变化率</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="time" 
                        label={{ value: '时间 (s)', position: 'insideBottom', offset: -5 }}
                      />
                      <YAxis 
                        label={{ value: 'ΔT/Δt', angle: -90, position: 'insideLeft' }}
                      />
                      <Tooltip 
                        formatter={(value: number) => [value.toFixed(4), 'ΔT/Δt']}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="deltaRatio" 
                        stroke="#10b981" 
                        strokeWidth={2}
                        name="ΔT/Δt"
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* 预测结果 */}
            {results && (
              <div className="bg-white rounded-lg shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">预测结果</h3>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <h4 className="font-medium text-blue-900 mb-2">导热系数</h4>
                      <p className="text-2xl font-bold text-blue-600">
                        {results.lambda_predicted.toFixed(4)} W/(m·K)
                      </p>
                    </div>
                    
                    <div className="p-4 bg-green-50 rounded-lg">
                      <h4 className="font-medium text-green-900 mb-2">修正后 T2</h4>
                      <p className="text-2xl font-bold text-green-600">
                        {results.T2_corrected.toFixed(2)} °C
                      </p>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <h4 className="font-medium text-gray-900 mb-2">修正参数</h4>
                      <div className="space-y-1 text-sm">
                        <p>参数 a: {results.correction_params.a.toFixed(4)}</p>
                        <p>参数 b: {results.correction_params.b.toFixed(4)}</p>
                      </div>
                    </div>
                    
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <h4 className="font-medium text-gray-900 mb-2">中间计算值</h4>
                      <div className="space-y-1 text-sm">
                        {Object.entries(results.intermediate_values).map(([key, value]) => (
                          <p key={key}>
                            {key}: {typeof value === 'number' ? value.toFixed(4) : value}
                          </p>
                        ))}
                      </div>
                    </div>
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
                    <strong>稳态法实验</strong>通过测量材料在稳态传热条件下的温度分布，计算材料的导热系数。
                  </p>
                  <p className="mb-3">
                    <strong>物理原理：</strong>基于傅里叶导热定律，在稳态条件下，热流密度与温度梯度成正比，比例系数即为导热系数。
                  </p>
                  <p className="mb-3">
                    <strong>应用场景：</strong>建筑材料导热性能测试、保温材料评估、工业材料热物性表征等。
                  </p>
                  <p>
                    <strong>参数说明：</strong>
                  </p>
                  <ul className="list-disc list-inside mt-2 space-y-1">
                    <li>T1: 热端温度，材料热端表面温度</li>
                    <li>T2: 冷端温度，材料冷端表面温度</li>
                    <li>仿真时长: 达到稳态所需的时间</li>
                    <li>噪声水平: 模拟实际测量中的随机误差</li>
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}