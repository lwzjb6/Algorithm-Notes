<font face="楷体" size = 3>

<center><font face="楷体" size=6, color='red'> BFS </font> </center>

**适用场景：找最短路，最少变换次数等**

### 经典问题：迷宫找最短路
```
问从左上角走到右下角最少需要几步？
输入
5 5     迷宫大小
0 1 0 0 0   0表示可走，1表示障碍物
0 1 0 1 0
0 0 0 0 0
0 1 1 1 0
0 0 0 1 0
输出
8
```

```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 110;
int n, m;
int a[N][N], vis[N][N];
struct node{
    int x, y;
    int step;
};
int dir[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
bool check(int x, int y) {
    if(x < 0 || x >= n || y < 0 || y >= m || vis[x][y] || a[x][y]) return 0;
    return 1;
}
int main()
{
    
    cin >> n >> m;
    for(int i = 0; i < n; i++) 
        for(int j = 0; j < m; j++)
            cin >> a[i][j];
    queue<node>q;
    q.push({0, 0, 0});
    while(q.size()) {
        auto e = q.front();
        q.pop();
        
        if(e.x == n - 1 && e.y == m - 1){
            cout << e.step << endl;
            break;
        }
        
        for(int i = 0; i < 4; i ++) {
            int nx = e.x + dir[i][0];
            int ny = e.y + dir[i][1];
            if(check(nx, ny)) {
                q.push({nx, ny, e.step + 1});
                vis[nx][ny] = 1;
            }
        }
    }
    
    return 0;
}
```