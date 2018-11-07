#ifndef CAFFE_SOLVER_HPP_
#define CAFFE_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"
#include "caffe/util/benchmark.hpp"

//参考：https://blog.csdn.net/fengbingchun/article/details/64444599

namespace caffe {

/**
  * @brief Enumeration of actions that a client of the Solver may request by
  * implementing the Solver's action request function, which a
  * client may optionally provide in order to request early termination
  * or saving a snapshot without exiting. In the executable caffe, this
  * mechanism is used to allow the snapshot to be saved when stopping
  * execution with a SIGINT (Ctrl-C).
  */
  namespace SolverAction {
    enum Enum {
      NONE = 0,  // Take no special action.
      STOP = 1,  // Stop training. snapshot_after_train controls whether a
                 // snapshot is created.
      SNAPSHOT = 2  // Take a snapshot, and keep training.
    };
  }

/**
 * @brief Type of a function that returns a Solver Action enumeration.
 */
typedef boost::function<SolverAction::Enum()> ActionCallback;

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
template <typename Dtype>
class Solver { // Solver模板类，虚基类
 public:
  explicit Solver(const SolverParameter& param); //显示构造函数,内部会调用Init函数
  explicit Solver(const string& param_file);
 
  // 成员变量赋值，包括param_、iter_、current_step_,并调用InitTrainNet和InitTestNets函数 
  void Init(const SolverParameter& param);
  
  // 为成员变量net_赋值
  void InitTrainNet();
  
  // 为成员变量test_nets_赋值
  void InitTestNets();

  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
  void SetActionFunction(ActionCallback func);
  SolverAction::Enum GetRequestedAction();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  // 依次调用函数Restore、Step、Snapshot，然后执行net_的前向传播函数ForwardPrefilled，最后调用TestAll函数
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string& resume_file) { Solve(resume_file.c_str()); }
  // 反复执行net前向传播反向传播计算,期间会调用函数TestAll、ApplyUpdate、Snapshot及类Callback两个成员函数
  void Step(int iters);
  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  void Restore(const char* resume_file); // 加载已有的模型
  
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  // 快照，内部会调用SnapshotToBinaryProto或SnapshotToHDF5、SnapshotSolverState函数
  void Snapshot();
  virtual ~Solver() {} // 虚析构函数
  // 获得slover parameter
  inline const SolverParameter& param() const { return param_; }
  // 获得train Net
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  // 获得test Net
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  // 获得当前的迭代数
  int iter() const { return iter_; }

  // Invoked at specific points during an iteration
  class Callback { // 内部Callback类，仅在多卡GPU模式下使用
   protected:
    virtual void on_start() = 0;
    virtual void on_gradients_ready() = 0;

    template <typename T>
    friend class Solver;
  };
  // 获得Callback
  const vector<Callback*>& callbacks() const { return callbacks_; }
  void add_callback(Callback* value) {// 添加一个Callback
    callbacks_.push_back(value);
  }

  void CheckSnapshotWritePermissions();
  /**
   * @brief Returns the solver type.
   */
  virtual inline const char* type() const { return ""; }

  // Make and apply the update value for the current iteration.
  virtual void ApplyUpdate() = 0;

 protected:
  // 更新net的权值和偏置
  string SnapshotFilename(const string& extension);
  string SnapshotToBinaryProto();
  string SnapshotToHDF5();
  // The test routine
  void TestAll();
  void Test(const int test_net_id = 0);
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
  void DisplayOutputBlobs(const int net_id);
  void UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss);

  // Caffe中类的成员变量名都带有后缀"_"，这样就容易区分临时变量和类成员变量
  SolverParameter param_; // solver parameter
  int iter_; //当前的迭代数
  int current_step_;
  shared_ptr<Net<Dtype> > net_; // train net
  vector<shared_ptr<Net<Dtype> > > test_nets_; // test net
  vector<Callback*> callbacks_; // Callback
  vector<Dtype> losses_;
  Dtype smoothed_loss_;

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;

  // True iff a request to stop early was received.
  bool requested_early_exit_;

  // Timing information, handy to tune e.g. nbr of GPUs
  Timer iteration_timer_;
  float iterations_last_;
  
  // 禁止使用Solver类的拷贝和赋值操作
  DISABLE_COPY_AND_ASSIGN(Solver);
};

}  // namespace caffe

#endif  // CAFFE_SOLVER_HPP_
