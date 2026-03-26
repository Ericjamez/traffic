// 简化版JavaScript代码 - 只保留完整时间点选择器
document.addEventListener('DOMContentLoaded', function() {
    // 全局变量
    let currentTrafficData = [];
    let chart = null;
    let currentChartType = 'bar';
    let currentSortOrder = 'desc';
    let autoRefreshInterval = null;
    let isAutoRefreshEnabled = false;
    
    // 初始化ECharts实例
    const chartDom = document.getElementById('trafficFlowChart');
    chart = echarts.init(chartDom);
    
    // 初始化完整时间点选择器
    initExactTimeSelector();
    
    // 初始化图表类型切换
    initChartTypeButtons();
    
    // 初始化排序方式切换
    initSortSelector();
    
    // 初始化自动刷新按钮
    initAutoRefreshButton();
    
    // 窗口大小变化时重绘图表
    window.addEventListener('resize', function() {
        if (chart) {
            chart.resize();
        }
    });
});

// 初始化完整时间点选择器
function initExactTimeSelector() {
    const exactTimeSelector = document.getElementById('exactTimeSelector');
    
    // 获取可用的完整时间点
    fetch('/api/traffic_available_times')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.available_times && data.available_times.length > 0) {
                // 清空现有选项
                exactTimeSelector.innerHTML = '<option value="">请选择时间点</option>';
                
                // 添加可用的时间点选项（按时间倒序排列，最新的在前面）
                const sortedTimes = [...data.available_times].reverse();
                sortedTimes.forEach(time => {
                    const option = document.createElement('option');
                    option.value = time;
                    option.textContent = time;
                    exactTimeSelector.appendChild(option);
                });
                
                // 设置默认值为最新时间点
                if (sortedTimes.length > 0) {
                    exactTimeSelector.value = sortedTimes[0];
                    // 自动加载最新数据
                    setTimeout(() => loadHistoryData(), 500);
                }
            } else {
                exactTimeSelector.innerHTML = '<option value="">暂无可用时间点</option>';
            }
        })
        .catch(error => {
            console.error('获取时间点数据错误:', error);
            exactTimeSelector.innerHTML = '<option value="">获取数据失败</option>';
        });
}

// 初始化图表类型按钮
function initChartTypeButtons() {
    document.querySelectorAll('.chart-btn[data-type]').forEach(btn => {
        btn.addEventListener('click', function() {
            // 移除所有按钮的active类
            document.querySelectorAll('.chart-btn[data-type]').forEach(b => {
                b.classList.remove('active');
            });
            
            // 为当前按钮添加active类
            this.classList.add('active');
            
            // 更新图表类型
            currentChartType = this.getAttribute('data-type');
            updateChartWithData(currentTrafficData);
        });
    });
}

// 初始化排序方式选择器
function initSortSelector() {
    const sortSelector = document.querySelector('.sort-select');
    sortSelector.addEventListener('change', function() {
        currentSortOrder = this.value;
        updateChartWithData(currentTrafficData);
    });
}

// 初始化自动刷新按钮
function initAutoRefreshButton() {
    const autoRefreshBtn = document.getElementById('autoRefreshBtn');
    autoRefreshBtn.addEventListener('click', toggleAutoRefresh);
}

// 切换自动刷新
function toggleAutoRefresh() {
    const autoRefreshBtn = document.getElementById('autoRefreshBtn');
    isAutoRefreshEnabled = !isAutoRefreshEnabled;
    
    if (isAutoRefreshEnabled) {
        // 启用自动刷新
        autoRefreshBtn.innerHTML = '<i class="fas fa-stop-circle"></i> 停止刷新';
        autoRefreshBtn.classList.add('active');
        
        // 每30秒自动刷新一次
        autoRefreshInterval = setInterval(() => {
            loadHistoryData();
        }, 30000);
        
        // 显示提示
        showToast('自动刷新已启用，每30秒刷新一次');
    } else {
        // 停止自动刷新
        autoRefreshBtn.innerHTML = '<i class="fas fa-redo"></i> 自动刷新';
        autoRefreshBtn.classList.remove('active');
        
        if (autoRefreshInterval) {
            clearInterval(autoRefreshInterval);
            autoRefreshInterval = null;
        }
        
        // 显示提示
        showToast('自动刷新已停止');
    }
}

// 加载历史数据
function loadHistoryData() {
    const exactTimeSelector = document.getElementById('exactTimeSelector');
    const loadBtn = document.getElementById('loadDataBtn');
    
    const exactTime = exactTimeSelector.value;
    
    if (!exactTime) {
        showToast('请选择时间点', 'warning');
        return;
    }
    
    // 显示加载状态
    loadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 加载中...';
    loadBtn.disabled = true;
    
    // 显示加载遮罩
    showLoading();
    
    // 构建API URL
    const url = `/api/traffic_flow_history?exact_time=${encodeURIComponent(exactTime)}`;
    
    // 发送请求
    fetch(url)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 更新数据
                currentTrafficData = data.data;
                
                // 更新数据统计信息
                updateDataInfo(data);
                
                // 更新图表
                updateChartWithData(currentTrafficData);
                
                // 更新最后更新时间
                const timeText = data.selected_exact_time || exactTime;
                document.querySelector('.update-time').textContent = 
                    `最后更新时间：${timeText}`;
                
                // 显示成功提示
                showToast(`已加载 ${data.road_count} 条道路数据`, 'success');
            } else {
                showToast('加载数据失败：' + data.msg, 'error');
            }
        })
        .catch(error => {
            console.error('加载数据错误:', error);
            showToast('加载数据失败，请检查网络连接', 'error');
        })
        .finally(() => {
            // 恢复按钮状态
            loadBtn.innerHTML = '<i class="fas fa-sync-alt"></i> 加载数据';
            loadBtn.disabled = false;
            
            // 隐藏加载遮罩
            hideLoading();
        });
}

// 显示加载遮罩
function showLoading() {
    const chartContainer = document.getElementById('trafficFlowChart');
    if (!chartContainer.querySelector('.loading-overlay')) {
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.innerHTML = '<div class="loading-spinner"></div>';
        chartContainer.style.position = 'relative';
        chartContainer.appendChild(overlay);
    }
}

// 隐藏加载遮罩
function hideLoading() {
    const chartContainer = document.getElementById('trafficFlowChart');
    const overlay = chartContainer.querySelector('.loading-overlay');
    if (overlay) {
        chartContainer.removeChild(overlay);
    }
}

// 更新数据统计信息
function updateDataInfo(data) {
    const dataInfo = document.getElementById('dataInfo');
    const roadCount = document.getElementById('roadCount');
    const recordCount = document.getElementById('recordCount');
    const selectedTimeInfo = document.getElementById('selectedTimeInfo');
    
    if (data.data.length > 0) {
        roadCount.textContent = data.road_count;
        recordCount.textContent = data.record_count;
        
        const timeText = `<i class="fas fa-clock"></i> ${data.selected_exact_time || '未知时间'}`;
        selectedTimeInfo.innerHTML = timeText;
        dataInfo.style.display = 'flex';
    } else {
        dataInfo.style.display = 'none';
        // 显示无数据提示
        showNoDataMessage();
    }
}

// 显示无数据提示
function showNoDataMessage() {
    const chartContainer = document.getElementById('trafficFlowChart');
    const existingMessage = chartContainer.querySelector('.no-data-message');
    
    if (!existingMessage) {
        const message = document.createElement('div');
        message.className = 'no-data-message';
        message.innerHTML = `
            <i class="fas fa-database"></i>
            <p>该时间点没有数据</p>
            <small>请选择其他时间点</small>
        `;
        chartContainer.appendChild(message);
    }
}

// 使用新数据更新图表
function updateChartWithData(data) {
    // 移除无数据提示
    const chartContainer = document.getElementById('trafficFlowChart');
    const noDataMessage = chartContainer.querySelector('.no-data-message');
    if (noDataMessage) {
        chartContainer.removeChild(noDataMessage);
    }
    
    if (data.length === 0) {
        showNoDataMessage();
        return;
    }
    
    // 准备图表数据
    const chartData = prepareChartData(data, currentSortOrder);
    const option = getChartOption(currentChartType, chartData);
    chart.setOption(option, true);
}

// 准备图表数据
function prepareChartData(data, sortOrder = 'desc') {
    let sortedData = [...data];
    
    if (sortOrder === 'desc') {
        sortedData.sort((a, b) => b.flow - a.flow);
    } else if (sortOrder === 'asc') {
        sortedData.sort((a, b) => a.flow - b.flow);
    } else if (sortOrder === 'name') {
        sortedData.sort((a, b) => a.road_name.localeCompare(b.road_name, 'zh-CN'));
    }
    
    return {
        roads: sortedData.map(item => item.road_name),
        flows: sortedData.map(item => item.flow),
        colors: sortedData.map(item => item.flow_color),
        levels: sortedData.map(item => item.flow_level),
        percentages: sortedData.map(item => item.flow_percentage),
        formatted: sortedData.map(item => item.flow_formatted)
    };
}

// 获取图表配置选项
function getChartOption(type, chartData) {
    const baseOption = {
        backgroundColor: '#fff',
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            },
            formatter: function(params) {
                const data = params[0];
                const index = data.dataIndex;
                return `
                    <div style="font-weight: bold; margin-bottom: 5px;">${chartData.roads[index]}</div>
                    <div>车流量：${chartData.formatted[index]}</div>
                    <div>流量等级：<span style="color: ${chartData.colors[index]}">${chartData.levels[index]}</span></div>
                    <div>占比：${chartData.percentages[index]}%</div>
                `;
            }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '10%',
            top: '10%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: chartData.roads,
            axisLabel: {
                rotate: 30,
                fontSize: 12
            },
            axisLine: {
                lineStyle: {
                    color: '#ccc'
                }
            }
        },
        yAxis: {
            type: 'value',
            name: '车流量 (辆/小时)',
            nameTextStyle: {
                fontSize: 12,
                padding: [0, 0, 0, 10]
            },
            axisLine: {
                lineStyle: {
                    color: '#ccc'
                }
            },
            splitLine: {
                lineStyle: {
                    type: 'dashed',
                    color: '#e0e0e0'
                }
            }
        }
    };
    
    if (type === 'bar') {
        return {
            ...baseOption,
            title: {
                text: '路段车流量统计',
                left: 'center',
                textStyle: {
                    fontSize: 16,
                    fontWeight: 'normal'
                }
            },
            series: [{
                name: '车流量',
                type: 'bar',
                data: chartData.flows.map((flow, index) => ({
                    value: flow,
                    itemStyle: {
                        color: chartData.colors[index]
                    }
                })),
                barWidth: '60%',
                label: {
                    show: true,
                    position: 'top',
                    formatter: '{c}',
                    fontSize: 12
                },
                itemStyle: {
                    borderRadius: [4, 4, 0, 0]
                }
            }]
        };
    } else if (type === 'line') {
        return {
            ...baseOption,
            title: {
                text: '路段车流量趋势',
                left: 'center',
                textStyle: {
                    fontSize: 16,
                    fontWeight: 'normal'
                }
            },
            series: [{
                name: '车流量',
                type: 'line',
                data: chartData.flows,
                smooth: true,
                symbol: 'circle',
                symbolSize: 8,
                lineStyle: {
                    width: 3,
                    color: '#007bff'
                },
                itemStyle: {
                    color: '#007bff',
                    borderColor: '#fff',
                    borderWidth: 2
                },
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: 'rgba(0, 123, 255, 0.3)' },
                        { offset: 1, color: 'rgba(0, 123, 255, 0.1)' }
                    ])
                },
                label: {
                    show: true,
                    position: 'top',
                    formatter: '{c}',
                    fontSize: 12
                }
            }]
        };
    } else if (type === 'pie') {
        return {
            title: {
                text: '路段车流量分布',
                left: 'center',
                textStyle: {
                    fontSize: 16,
                    fontWeight: 'normal'
                }
            },
            tooltip: {
                trigger: 'item',
                formatter: function(params) {
                    return `
                        <div style="font-weight: bold; margin-bottom: 5px;">${params.name}</div>
                        <div>车流量：${params.data.formatted}</div>
                        <div>占比：${params.percent}%</div>
                        <div>流量等级：<span style="color: ${params.color}">${params.data.level}</span></div>
                    `;
                }
            },
            legend: {
                orient: 'vertical',
                left: 'left',
                top: 'center',
                data: chartData.roads
            },
            series: [{
                name: '车流量',
                type: 'pie',
                radius: ['40%', '70%'],
                center: ['60%', '50%'],
                data: chartData.roads.map((road, index) => ({
                    name: road,
                    value: chartData.flows[index],
                    formatted: chartData.formatted[index],
                    level: chartData.levels[index],
                    itemStyle: {
                        color: chartData.colors[index]
                    }
                })),
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowOffsetX: 0,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                },
                label: {
                    formatter: '{b}: {d}%'
                }
            }]
        };
    }
    
    return baseOption;
}

// 显示提示消息
function showToast(message, type = 'info') {
    // 移除现有的提示
    const existingToast = document.querySelector('.toast-message');
    if (existingToast) {
        existingToast.remove();
    }
    
    // 创建新的提示
    const toast = document.createElement('div');
    toast.className = `toast-message toast-${type}`;
    toast.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    
    // 添加到页面
    document.body.appendChild(toast);
    
    // 显示提示
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);
    
    // 3秒后自动隐藏
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 300);
    }, 3000);
}

// 添加提示样式
const toastStyle = document.createElement('style');
toastStyle.textContent = `
    .toast-message {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 8px;
        background-color: #fff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        gap: 10px;
        z-index: 9999;
        transform: translateX(100%);
        opacity: 0;
        transition: transform 0.3s ease, opacity 0.3s ease;
    }
    .toast-message.show {
        transform: translateX(0);
        opacity: 1;
    }
    .toast-success {
        border-left: 4px solid #28a745;
    }
    .toast-success i {
        color: #28a745;
    }
    .toast-error {
        border-left: 4px solid #dc3545;
    }
    .toast-error i {
        color: #dc3545;
    }
    .toast-warning {
        border-left: 4px solid #ffc107;
    }
    .toast-warning i {
        color: #ffc107;
    }
    .toast-info {
        border-left: 4px solid #007bff;
    }
    .toast-info i {
        color: #007bff;
    }
`;
document.head.appendChild(toastStyle);