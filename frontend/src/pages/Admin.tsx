import React, { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Key, Upload, Download, Play, Pause, RefreshCw, Database, Settings, FileText, AlertCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { adminAPI, ModelInfo, TrainingRequest, TrainingResponse } from '@/lib/api';

interface TrainingTask {
  task_id: string;
  model_type: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  start_time: string;
  end_time?: string;
  config: any;
  error?: string;
}

interface TrainingHistory {
  task_id: string;
  model_type: string;
  status: string;
  start_time: string;
  end_time?: string;
  duration?: number;
  final_metrics?: any;
}

export default function Admin() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [adminToken, setAdminToken] = useState('');
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [trainingTasks, setTrainingTasks] = useState<TrainingTask[]>([]);
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistory[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'models' | 'training' | 'history' | 'settings'>('models');
  const [showTrainingConfig, setShowTrainingConfig] = useState<string | null>(null);
  const [trainingConfig, setTrainingConfig] = useState({
    epochs: 100,
    batch_size: 32,
    learning_rate: 0.001,
    validation_split: 0.2
  });

  // 检查本地存储的token
  useEffect(() => {
    const token = localStorage.getItem('admin_token');
    if (token) {
      setAdminToken(token);
      setIsAuthenticated(true);
      loadData();
    }
  }, []);

  const handleLogin = async () => {
    if (!adminToken.trim()) {
      setError('请输入管理员令牌');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      // 保存token到localStorage
      localStorage.setItem('admin_token', adminToken);
      
      // 尝试获取模型列表来验证token
      await adminAPI.getModels();
      
      setIsAuthenticated(true);
      await loadData();
    } catch (err: any) {
      setError(err.response?.data?.detail || '认证失败');
      localStorage.removeItem('admin_token');
      setIsAuthenticated(false);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('admin_token');
    setIsAuthenticated(false);
    setAdminToken('');
    setModels([]);
    setTrainingTasks([]);
    setTrainingHistory([]);
  };

  const loadData = async () => {
    try {
      setLoading(true);
      const [modelsData, historyData] = await Promise.all([
        adminAPI.getModels(),
        adminAPI.getTrainingHistory(),
      ]);
      
      setModels(modelsData);
      setTrainingHistory(historyData);
    } catch (err: any) {
      setError(err.response?.data?.detail || '加载数据失败');
    } finally {
      setLoading(false);
    }
  };

  const startTraining = async (modelType: 'steady' | 'quasi') => {
    try {
      setLoading(true);
      setError(null);
      
      const request: TrainingRequest = {
        model_type: modelType,
        epochs: trainingConfig.epochs,
        batch_size: trainingConfig.batch_size,
        learning_rate: trainingConfig.learning_rate,
        validation_split: trainingConfig.validation_split,
      };
      
      const response = await adminAPI.startTraining(request);
      
      // 添加到训练任务列表
      const newTask: TrainingTask = {
        task_id: response.task_id,
        model_type: modelType,
        status: 'pending',
        progress: 0,
        start_time: new Date().toISOString(),
        config: {
          epochs: request.epochs,
          batch_size: request.batch_size,
          learning_rate: request.learning_rate,
          validation_split: request.validation_split,
        },
      };
      
      setTrainingTasks(prev => [newTask, ...prev]);
      
      // 开始轮询任务状态
      pollTrainingStatus(response.task_id);
    } catch (err: any) {
      setError(err.response?.data?.detail || '启动训练失败');
    } finally {
      setLoading(false);
    }
  };

  const pollTrainingStatus = useCallback(async (taskId: string) => {
    try {
      const status = await adminAPI.getTrainingStatus(taskId);
      
      setTrainingTasks(prev => 
        prev.map(task => 
          task.task_id === taskId 
            ? { ...task, ...status }
            : task
        )
      );
      
      // 如果任务还在运行，继续轮询
      if (status.status === 'running' || status.status === 'pending') {
        setTimeout(() => pollTrainingStatus(taskId), 2000);
      } else {
        // 任务完成，刷新数据并停止轮询
        await loadData();
      }
    } catch (err) {
      console.error('轮询训练状态失败:', err);
      // 如果获取状态失败，停止轮询
      setTrainingTasks(prev => 
        prev.map(task => 
          task.task_id === taskId 
            ? { ...task, status: 'failed', error: '获取状态失败' }
            : task
        )
      );
    }
  }, [loadData]);

  // 自动轮询效果
  useEffect(() => {
    const runningTasks = trainingTasks.filter(task => 
      task.status === 'running' || task.status === 'pending'
    );
    
    runningTasks.forEach(task => {
      const intervalId = setInterval(() => {
        pollTrainingStatus(task.task_id);
      }, 3000);
      
      // 清理定时器
      setTimeout(() => clearInterval(intervalId), 300000); // 5分钟后自动停止轮询
    });
  }, [trainingTasks, pollTrainingStatus]);

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  // 登录界面
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="max-w-md w-full">
          <div className="bg-white rounded-lg shadow-sm p-8">
            <div className="text-center mb-8">
              <Key className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <h1 className="text-2xl font-bold text-gray-900">管理员登录</h1>
              <p className="text-gray-600 mt-2">请输入管理员令牌以访问管理功能</p>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  管理员令牌
                </label>
                <input
                  type="password"
                  value={adminToken}
                  onChange={(e) => setAdminToken(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="输入管理员令牌"
                  onKeyPress={(e) => e.key === 'Enter' && handleLogin()}
                />
              </div>
              
              {error && (
                <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-sm text-red-600">{error}</p>
                </div>
              )}
              
              <button
                onClick={handleLogin}
                disabled={loading}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
              >
                {loading ? '验证中...' : '登录'}
              </button>
              
              <div className="text-center">
                <Link to="/" className="text-sm text-gray-600 hover:text-gray-900">
                  返回首页
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

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
              <h1 className="text-2xl font-bold text-gray-900">管理员控制台</h1>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={loadData}
                disabled={loading}
                className="flex items-center gap-2 px-3 py-2 text-gray-600 hover:text-gray-900"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                刷新
              </button>
              <button
                onClick={handleLogout}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
              >
                退出登录
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* 标签页导航 */}
        <div className="mb-8">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => setActiveTab('models')}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'models'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Database className="w-4 h-4 inline mr-2" />
                模型管理
              </button>
              <button
                onClick={() => setActiveTab('training')}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'training'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Settings className="w-4 h-4 inline mr-2" />
                训练任务
              </button>
              <button
                onClick={() => setActiveTab('history')}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'history'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <FileText className="w-4 h-4 inline mr-2" />
                训练历史
              </button>
              <button
                onClick={() => setActiveTab('settings')}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'settings'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Settings className="w-4 h-4 inline mr-2" />
                系统设置
              </button>
            </nav>
          </div>
        </div>

        {/* 错误提示 */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <p className="text-red-600">{error}</p>
            </div>
          </div>
        )}

        {/* 模型管理 */}
        {activeTab === 'models' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">已部署模型</h2>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        模型名称
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        版本
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        时间戳
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        文件大小
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        文件路径
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {models.map((model, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {model.name}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {model.version}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {new Date(model.timestamp).toLocaleString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {formatFileSize(model.size)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 font-mono">
                          {model.file_path}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* 训练任务 */}
        {activeTab === 'training' && (
          <div className="space-y-6">
            {/* 训练参数配置 */}
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">训练参数配置</h2>
              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">训练轮数</label>
                  <input
                    type="number"
                    value={trainingConfig.epochs}
                    onChange={(e) => setTrainingConfig(prev => ({ ...prev, epochs: parseInt(e.target.value) || 100 }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    min="1"
                    max="1000"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">批次大小</label>
                  <input
                    type="number"
                    value={trainingConfig.batch_size}
                    onChange={(e) => setTrainingConfig(prev => ({ ...prev, batch_size: parseInt(e.target.value) || 32 }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    min="1"
                    max="256"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">学习率</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={trainingConfig.learning_rate}
                    onChange={(e) => setTrainingConfig(prev => ({ ...prev, learning_rate: parseFloat(e.target.value) || 0.001 }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    min="0.0001"
                    max="1"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">验证集比例</label>
                  <input
                    type="number"
                    step="0.1"
                    value={trainingConfig.validation_split}
                    onChange={(e) => setTrainingConfig(prev => ({ ...prev, validation_split: parseFloat(e.target.value) || 0.2 }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    min="0.1"
                    max="0.5"
                  />
                </div>
              </div>
            </div>
            
            {/* 启动训练 */}
            <div className="bg-white rounded-lg shadow-sm p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">启动新训练</h2>
              </div>
              <div className="grid md:grid-cols-2 gap-4">
                <button
                  onClick={() => startTraining('steady')}
                  disabled={loading}
                  className="flex items-center justify-center gap-2 p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-500 hover:bg-blue-50 disabled:opacity-50"
                >
                  <Play className="w-5 h-5 text-blue-600" />
                  <div className="text-left">
                    <p className="font-medium text-gray-900">稳态法模型训练</p>
                    <p className="text-sm text-gray-500">重新训练稳态法神经网络模型</p>
                    <p className="text-xs text-gray-400 mt-1">
                      {trainingConfig.epochs}轮 | 批次:{trainingConfig.batch_size} | 学习率:{trainingConfig.learning_rate}
                    </p>
                  </div>
                </button>
                
                <button
                  onClick={() => startTraining('quasi')}
                  disabled={loading}
                  className="flex items-center justify-center gap-2 p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-green-500 hover:bg-green-50 disabled:opacity-50"
                >
                  <Play className="w-5 h-5 text-green-600" />
                  <div className="text-left">
                    <p className="font-medium text-gray-900">准稳态法模型训练</p>
                    <p className="text-sm text-gray-500">重新训练准稳态法多任务模型</p>
                    <p className="text-xs text-gray-400 mt-1">
                      {trainingConfig.epochs}轮 | 批次:{trainingConfig.batch_size} | 学习率:{trainingConfig.learning_rate}
                    </p>
                  </div>
                </button>
              </div>
            </div>

            {/* 当前训练任务 */}
            {trainingTasks.length > 0 && (
              <div className="bg-white rounded-lg shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">当前训练任务</h3>
                <div className="space-y-4">
                  {trainingTasks.map((task) => (
                    <div key={task.task_id} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <h4 className="font-medium text-gray-900">
                            {task.model_type === 'steady' ? '稳态法模型' : '准稳态法模型'}
                          </h4>
                          <span className={`px-2 py-1 text-xs rounded-full ${
                            task.status === 'running' ? 'bg-blue-100 text-blue-800' :
                            task.status === 'completed' ? 'bg-green-100 text-green-800' :
                            task.status === 'failed' ? 'bg-red-100 text-red-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {task.status === 'running' ? '运行中' :
                             task.status === 'completed' ? '已完成' :
                             task.status === 'failed' ? '失败' : '等待中'}
                          </span>
                        </div>
                        <p className="text-sm text-gray-500">
                          {new Date(task.start_time).toLocaleString()}
                        </p>
                      </div>
                      
                      {/* 进度条 - 显示所有任务的进度 */}
                      {(task.status === 'running' || task.status === 'pending') && (
                        <div className="mb-3">
                          <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
                            <span>训练进度</span>
                            <span className="font-medium">{task.progress || 0}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2.5">
                            <div 
                              className="bg-blue-600 h-2.5 rounded-full transition-all duration-500 ease-out"
                              style={{ width: `${task.progress || 0}%` }}
                            />
                          </div>
                          {/* 实时刷新指示器 */}
                          <div className="flex items-center gap-2 mt-2 text-xs text-gray-500">
                            <div className="animate-spin w-3 h-3 border border-blue-500 border-t-transparent rounded-full"></div>
                            <span>实时更新中...</span>
                          </div>
                        </div>
                      )}
                      
                      {/* 已完成任务显示最终进度 */}
                      {task.status === 'completed' && (
                        <div className="mb-3">
                          <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
                            <span>训练进度</span>
                            <span className="font-medium text-green-600">100%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2.5">
                            <div className="bg-green-600 h-2.5 rounded-full w-full" />
                          </div>
                        </div>
                      )}
                      
                      {task.error && (
                        <div className="mt-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                          <div className="flex items-start gap-2">
                            <AlertCircle className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
                            <p className="text-sm text-red-600">{task.error}</p>
                          </div>
                        </div>
                      )}
                      
                      <div className="mt-3 pt-2 border-t border-gray-100">
                        <p className="text-xs text-gray-500 font-mono">任务ID: {task.task_id}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* 训练历史 */}
        {activeTab === 'history' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">训练历史记录</h2>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        任务ID
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        模型类型
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        状态
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        开始时间
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        持续时间
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        最终指标
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {trainingHistory.map((record) => (
                      <tr key={record.task_id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 text-sm font-mono text-gray-900">
                          <div className="max-w-xs">
                            <p className="truncate" title={record.task_id}>{record.task_id}</p>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {record.model_type === 'steady' ? '稳态法' : '准稳态法'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`px-2 py-1 text-xs rounded-full ${
                            record.status === 'completed' ? 'bg-green-100 text-green-800' :
                            record.status === 'failed' ? 'bg-red-100 text-red-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {record.status === 'completed' ? '已完成' :
                             record.status === 'failed' ? '失败' : record.status}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {new Date(record.start_time).toLocaleString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {record.duration ? formatDuration(record.duration) : '-'}
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-500">
                          {record.final_metrics ? (
                            <div className="space-y-1">
                              <div>损失: {record.final_metrics.final_loss?.toFixed(4) || '-'}</div>
                              <div>准确率: {record.final_metrics.final_accuracy?.toFixed(3) || '-'}</div>
                            </div>
                          ) : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* 系统设置 */}
        {activeTab === 'settings' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">系统设置</h2>
              <div className="space-y-4">
                <div>
                  <h3 className="text-md font-medium text-gray-900 mb-2">系统信息</h3>
                  <div className="grid md:grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500">API地址:</span>
                      <span className="ml-2 font-mono">{import.meta.env.VITE_API_BASE_URL || 'http://localhost:8090'}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">当前时间:</span>
                      <span className="ml-2">{new Date().toLocaleString()}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">系统状态:</span>
                      <span className="ml-2 text-green-600">正常运行</span>
                    </div>
                  </div>
                </div>
                
                <div className="border-t pt-4">
                  <h3 className="text-md font-medium text-gray-900 mb-2">模型用途说明</h3>
                  <div className="space-y-3 text-sm text-gray-600">
                    <div className="bg-blue-50 p-4 rounded-lg">
                      <h4 className="font-medium text-blue-900 mb-2">稳态法模型</h4>
                      <p>用于预测材料的导热系数。通过输入材料的温度差和时间参数，模型可以准确预测导热系数值，适用于稳态传热条件下的材料热物性分析。</p>
                    </div>
                    <div className="bg-green-50 p-4 rounded-lg">
                      <h4 className="font-medium text-green-900 mb-2">准稳态法模型</h4>
                      <p>多任务神经网络模型，同时预测材料的导热系数λ和比热容c。适用于准稳态传热条件，能够提供更全面的材料热物性参数预测。</p>
                    </div>
                    <div className="bg-purple-50 p-4 rounded-lg">
                      <h4 className="font-medium text-purple-900 mb-2">应用场景</h4>
                      <ul className="list-disc list-inside space-y-1">
                        <li>材料科学研究：快速获取材料热物性参数</li>
                        <li>工程设计：为传热计算提供准确的材料参数</li>
                        <li>质量控制：批量检测材料热物性是否符合标准</li>
                        <li>教学实验：虚拟仿真实验，降低实验成本</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}