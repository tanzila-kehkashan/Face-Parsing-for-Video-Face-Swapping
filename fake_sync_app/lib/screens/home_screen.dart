import 'dart:convert';
import 'dart:io';
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_picker/file_picker.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import '../services/api_service.dart';
import '../services/navigation_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  final ApiService _apiService = ApiService();
  final ImagePicker _imagePicker = ImagePicker();
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  File? _sourceImage;
  File? _targetVideo;
  bool _isUploading = false;
  String _errorMessage = '';
  String _selectedMode = 'best';
  int _currentStep = 0;
  double _uploadProgress = 0.0;
  String _uploadStatus = '';

  // User data for drawer
  String _userName = 'Guest User';
  String _userEmail = 'guest@example.com';
  bool _isLoadingUser = true;

  // Processing variables
  String? _currentJobId;
  String _processingStatus = 'processing';
  String _processingMessage = 'Initializing...';
  Timer? _statusTimer;
  StreamSubscription<String>? _logSubscription;
  bool _isProcessing = false;
  DateTime? _processStartTime;

  // NSFW detection variables
  bool _nsfwDetectionEnabled = false;
  bool _showNsfwWarning = false;

  late AnimationController _animationController;
  late AnimationController _pulseController;
  late Animation<double> _fadeAnimation;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );
    _pulseController = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    );
    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeInOut),
    );
    _pulseAnimation = Tween<double>(begin: 0.8, end: 1.2).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
    _animationController.forward();

    // Load user data for drawer
    _loadUserData();

    // Check server capabilities including NSFW detection
    _checkServerCapabilities();
  }

  Future<void> _loadUserData() async {
    try {
      final user = _auth.currentUser;
      if (user != null) {
        final doc = await _firestore.collection('user').doc(user.uid).get();
        if (doc.exists) {
          setState(() {
            _userName = doc['name'] ?? 'User';
            _userEmail = user.email ?? 'guest@example.com';
            _isLoadingUser = false;
          });
        } else {
          setState(() {
            _userName = user.displayName ?? 'User';
            _userEmail = user.email ?? 'guest@example.com';
            _isLoadingUser = false;
          });
        }
      } else {
        setState(() => _isLoadingUser = false);
      }
    } catch (e) {
      setState(() {
        _userName = 'Error Loading';
        _userEmail = 'error@example.com';
        _isLoadingUser = false;
      });
    }
  }

  Future<void> _checkServerCapabilities() async {
    try {
      final capabilities = await _apiService.getServerCapabilities();
      if (mounted) {
        setState(() {
          _nsfwDetectionEnabled = capabilities['nsfw_detection_enabled'] ?? false;
        });
      }
    } catch (e) {
      print('Failed to check server capabilities: $e');
    }
  }

  @override
  void dispose() {
    _animationController.dispose();
    _pulseController.dispose();
    _statusTimer?.cancel();
    _logSubscription?.cancel();
    super.dispose();
  }

  // Drawer item builder
  Widget _buildDrawerItem({
    required IconData icon,
    required String title,
    required VoidCallback onTap,
    Color? textColor,
  }) {
    return ListTile(
      leading: Icon(
        icon,
        color: textColor ?? Colors.white.withOpacity(0.9),
        size: 22,
      ),
      title: Text(
        title,
        style: TextStyle(
          color: textColor ?? Colors.white.withOpacity(0.9),
          fontSize: 16,
          fontWeight: FontWeight.w500,
        ),
      ),
      onTap: onTap,
      contentPadding: const EdgeInsets.symmetric(horizontal: 24, vertical: 4),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(0),
      ),
    );
  }

  // Drawer builder
  Widget _buildDrawer() {
    return Drawer(
      backgroundColor: const Color(0xFF1E1E1E),
      child: Column(
        children: [
          Container(
            width: double.infinity,
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  Colors.purple.shade700,
                  Colors.purple.shade500,
                ],
              ),
            ),
            child: SafeArea(
              bottom: false,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(20, 20, 20, 30),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: 20),
                    CircleAvatar(
                      radius: 35,
                      backgroundColor: Colors.white.withOpacity(0.2),
                      child: const Icon(
                        Icons.person,
                        size: 40,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(height: 20),
                    _isLoadingUser
                        ? const Center(
                      child: CircularProgressIndicator(
                        color: Colors.white,
                        strokeWidth: 3,
                      ),
                    )
                        : Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          _userName,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          _userEmail,
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.8),
                            fontSize: 16,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                  ],
                ),
              ),
            ),
          ),
          Expanded(
            child: ListView(
              padding: const EdgeInsets.symmetric(vertical: 10),
              children: [
                _buildDrawerItem(
                  icon: Icons.person_outline,
                  title: 'Profile',
                  onTap: () {
                    Navigator.pop(context);
                    NavigationService.pushNamed('/profile');
                  },
                ),
                _buildDrawerItem(
                  icon: Icons.history,
                  title: 'History',
                  onTap: () {
                    Navigator.pop(context);
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('History feature coming soon!')),
                    );
                  },
                ),
                _buildDrawerItem(
                  icon: Icons.help_outline,
                  title: 'Help & Support',
                  onTap: () {
                    Navigator.pop(context);
                    NavigationService.goTohelp();
                  },
                ),
                const Divider(color: Colors.grey),
                _buildDrawerItem(
                  icon: Icons.logout,
                  title: 'Logout',
                  textColor: Colors.red.shade400,
                  onTap: () {
                    Navigator.pop(context);
                    NavigationService.goToLoginAfterLogout();
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _pickSourceImage() async {
    try {
      final XFile? pickedFile = await _imagePicker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 90,
      );

      if (pickedFile != null) {
        setState(() {
          _sourceImage = File(pickedFile.path);
          _errorMessage = '';
          if (_currentStep == 0) _currentStep = 1;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Error picking image: $e';
      });
    }
  }

  Future<void> _pickTargetVideo() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.video,
        allowMultiple: false,
      );

      if (result != null) {
        setState(() {
          _targetVideo = File(result.files.single.path!);
          _errorMessage = '';
          if (_currentStep == 1) _currentStep = 2;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Error picking video: $e';
      });
    }
  }

  Future<void> _uploadFilesAndProcess() async {
    if (_sourceImage == null || _targetVideo == null) {
      setState(() {
        _errorMessage = 'Please select both source image and target video';
      });
      return;
    }

    setState(() {
      _isUploading = true;
      _isProcessing = true;
      _errorMessage = '';
      _currentStep = 3;
      _uploadProgress = 0.0;
      _uploadStatus = 'Preparing upload...';
      _processStartTime = DateTime.now();
      _showNsfwWarning = false; // Reset NSFW warning
    });

    try {
      Map<String, dynamic>? settings;

      if (_selectedMode == 'best') {
        settings = {
          'face_enhancer_name': 'CodeFormer',
          'enable_face_parser': true,
          'enable_laplacian_blend': true,
        };
        print('DEBUG: Best mode - sending custom settings: $settings');
      } else {
        settings = null;
        print('DEBUG: Normal mode - using server defaults (no custom settings)');
      }

      // Create a custom upload function with progress tracking
      final String jobId = await _uploadWithProgress(
        _sourceImage!,
        _targetVideo!,
        settings != null ? json.encode(settings) : null,
      );

      if (!mounted) return;

      setState(() {
        _currentJobId = jobId;
        _isUploading = false;
        _processingMessage = 'Processing started successfully...';
      });

      // Start processing monitoring instead of navigating away
      _startProcessingMonitoring(jobId);

    } on NSFWContentException catch (e) {
      // Handle NSFW content specifically
      if (!mounted) return;
      setState(() {
        _errorMessage = e.toString();
        _currentStep = 2;
        _uploadProgress = 0.0;
        _uploadStatus = '';
        _isUploading = false;
        _isProcessing = false;
        _showNsfwWarning = true; // Show NSFW-specific warning
      });

      // Show specific NSFW error dialog
      _showNsfwErrorDialog(e.toString());

    } catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Upload failed: $e';
        _currentStep = 2;
        _uploadProgress = 0.0;
        _uploadStatus = '';
        _isUploading = false;
        _isProcessing = false;
      });
    }
  }

  void _showNsfwErrorDialog(String message) {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF2D2D2D),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        title: Row(
          children: [
            Icon(
              Icons.security,
              color: Colors.red.shade400,
              size: 28,
            ),
            const SizedBox(width: 12),
            const Text(
              'Content Policy Violation',
              style: TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              message,
              style: const TextStyle(
                color: Colors.grey,
                fontSize: 16,
              ),
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.orange.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: Colors.orange.withOpacity(0.3),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Content Safety Guidelines:',
                    style: TextStyle(
                      color: Colors.orange.shade300,
                      fontWeight: FontWeight.w600,
                      fontSize: 14,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '• Use appropriate, family-friendly images and videos\n'
                        '• Avoid content that may be considered inappropriate\n'
                        '• Ensure all content complies with platform policies\n'
                        '• Try different images if content is blocked',
                    style: TextStyle(
                      color: Colors.orange.shade200,
                      fontSize: 12,
                      height: 1.4,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text(
              'Choose Different Content',
              style: TextStyle(
                color: Colors.purple,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _startProcessingMonitoring(String jobId) {
    // Start pulse animation for processing
    _pulseController.repeat(reverse: true);

    // Start log streaming
    _logSubscription = _apiService.streamLogs(jobId).listen(
          (logLine) {
        if (mounted) {
          setState(() {
            if (logLine.isNotEmpty && !logLine.startsWith('Error')) {
              _processingMessage = logLine;
            }
          });
        }
      },
      onError: (error) {
        if (mounted) {
          setState(() {
            _processingMessage = 'Log streaming error: $error';
          });
        }
      },
      onDone: () {
        // Log stream ended
      },
    );

    // Start status checking
    _statusTimer = Timer.periodic(const Duration(seconds: 10), (timer) {
      _checkJobStatus(jobId);
    });

    // Initial status check
    _checkJobStatus(jobId);
  }

  Future<void> _checkJobStatus(String jobId) async {
    if (!mounted) return;

    try {
      final status = await _apiService.checkStatus(jobId);

      if (!mounted) return;

      setState(() {
        _processingStatus = status;
      });

      if (status == 'completed') {
        _statusTimer?.cancel();
        _logSubscription?.cancel();
        _pulseController.stop();

        setState(() {
          _processingMessage = 'Face swap completed! 🎉';
          _isProcessing = false;
        });

        // Navigate to result screen after a brief delay
        await Future.delayed(const Duration(seconds: 2));
        if (mounted) {
          NavigationService.goToResult(jobId);
        }
      } else if (status == 'failed') {
        _statusTimer?.cancel();
        _logSubscription?.cancel();
        _pulseController.stop();

        setState(() {
          _errorMessage = 'Face swap processing failed. Please try again.';
          _processingMessage = 'Processing failed ❌';
          _isProcessing = false;
          _currentStep = 2; // Go back to mode selection
        });
      } else if (status == 'cancelled') {
        _statusTimer?.cancel();
        _logSubscription?.cancel();
        _pulseController.stop();

        setState(() {
          _processingMessage = 'Processing cancelled by user';
          _isProcessing = false;
          _currentStep = 2; // Go back to mode selection
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Failed to check status: $e';
        });
      }
    }
  }

  Future<void> _cancelProcessing() async {
    try {
      if (_isUploading) {
        // Cancel the ongoing upload
        _apiService.cancelCurrentUpload();

        setState(() {
          _isUploading = false;
          _isProcessing = false;
          _uploadProgress = 0.0;
          _uploadStatus = '';
          _processingMessage = 'Upload cancelled by user';
          _currentStep = 2; // Go back to mode selection
          _currentJobId = null;
        });

        NavigationService.showSuccessSnackBar('Upload cancelled successfully');
        return;
      }

      // If we have a job ID, cancel on server
      if (_currentJobId != null) {
        final result = await _apiService.cancelJob(_currentJobId!);

        if (result['success']) {
          _statusTimer?.cancel();
          _logSubscription?.cancel();
          _pulseController.stop();

          setState(() {
            _processingMessage = 'Cancellation requested...';
            _processingStatus = 'cancelled';
          });

          NavigationService.showSuccessSnackBar('Job cancelled successfully');

          // Reset to mode selection after brief delay
          await Future.delayed(const Duration(seconds: 2));
          if (mounted) {
            setState(() {
              _isProcessing = false;
              _currentStep = 2;
              _currentJobId = null;
            });
          }
        } else {
          NavigationService.showErrorSnackBar(result['message'] ?? 'Failed to cancel job');
        }
      }
    } catch (e) {
      NavigationService.showErrorSnackBar('Failed to cancel: $e');
    }
  }

  Future<String> _uploadWithProgress(File sourceImage, File targetVideo, String? settings) async {
    print('Starting file upload...');
    print('Source image size: ${await sourceImage.length()} bytes');
    print('Target video size: ${await targetVideo.length()} bytes');

    setState(() {
      _uploadProgress = 0.1;
      _uploadStatus = 'Validating files...';
    });

    await Future.delayed(const Duration(milliseconds: 500));

    setState(() {
      _uploadProgress = 0.3;
      _uploadStatus = 'Uploading files...';
    });

    final jobId = await _apiService.uploadFiles(sourceImage, targetVideo, settings);

    setState(() {
      _uploadProgress = 1.0;
      _uploadStatus = 'Upload complete!';
    });

    return jobId;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: _scaffoldKey, // Added scaffold key
      drawer: _buildDrawer(),
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.menu, color: Colors.white),
          onPressed: () {
            _scaffoldKey.currentState?.openDrawer();
          },
        ),
        automaticallyImplyLeading: false,
        title: LayoutBuilder(
          builder: (context, constraints) {
            final screenWidth = MediaQuery.of(context).size.width;
            final isWideScreen = screenWidth > 400;

            return Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (isWideScreen) ...[
                  Icon(
                    Icons.face_retouching_natural,
                    color: Colors.white,
                    size: 28,
                  ),
                  const SizedBox(width: 8),
                ],
                Flexible(
                  child: Text(
                    'FAKESYNC STUDIO',
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      fontSize: isWideScreen ? 20 : 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            );
          },
        ),
        actions: [
          LayoutBuilder(
            builder: (context, constraints) {
              final screenWidth = MediaQuery.of(context).size.width;
              if (screenWidth > 350) {
                return IconButton(
                  icon: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(Icons.home, size: 20),
                  ),
                  onPressed: () async {
                    if (_isProcessing || _isUploading) {
                      bool shouldReset = await showDialog<bool>(
                        context: context,
                        builder: (context) => AlertDialog(
                          backgroundColor: const Color(0xFF2D2D2D),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16),
                          ),
                          title: const Text(
                            'Cancel Face Swap?',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          content: const Text(
                            'Your face swap is currently in progress. Going home will cancel the process and you\'ll lose your work. Are you sure?',
                            style: TextStyle(
                              color: Colors.grey,
                              fontSize: 16,
                            ),
                          ),
                          actions: [
                            TextButton(
                              onPressed: () => Navigator.pop(context, false),
                              child: const Text(
                                'Keep Processing',
                                style: TextStyle(color: Colors.purple),
                              ),
                            ),
                            TextButton(
                              onPressed: () => Navigator.pop(context, true),
                              child: const Text(
                                'Cancel & Go Home',
                                style: TextStyle(
                                  color: Colors.red,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ) ?? false;

                      if (!shouldReset) return;

                      if (_isUploading) {
                        _apiService.cancelCurrentUpload();
                      } else if (_currentJobId != null) {
                        await _cancelProcessing();
                      }
                    }

                    setState(() {
                      _currentStep = 0;
                      _sourceImage = null;
                      _targetVideo = null;
                      _errorMessage = '';
                      _isUploading = false;
                      _isProcessing = false;
                      _currentJobId = null;
                      _selectedMode = 'best';
                    });

                    _statusTimer?.cancel();
                    _logSubscription?.cancel();
                    _pulseController.stop();

                    NavigationService.showInfoSnackBar('Ready for new face swap');
                  },
                );
              } else {
                return const SizedBox.shrink();
              }
            },
          ),
          const SizedBox(width: 8),
        ],
        flexibleSpace: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Colors.purple.shade700,
                Colors.purple.shade500,
                Colors.purple.shade400,
              ],
            ),
          ),
        ),
      ),
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFF1E1E1E), Color(0xFF2D2D2D)],
          ),
        ),
        child: SafeArea(
          child: FadeTransition(
            opacity: _fadeAnimation,
            child: Column(
              children: [
                Expanded(child: _buildCurrentStep()),
                if (_errorMessage.isNotEmpty) _buildErrorMessage(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildCurrentStep() {
    switch (_currentStep) {
      case 0:
        return _buildUploadSourceStep();
      case 1:
        return _buildUploadTargetStep();
      case 2:
        return _buildModeSelectionStep();
      case 3:
        return _buildProcessingStep();
      default:
        return _buildUploadSourceStep();
    }
  }

  Widget _buildUploadSourceStep() {
    return LayoutBuilder(
        builder: (context, constraints) {
          return SingleChildScrollView(
            padding: const EdgeInsets.all(24.0),
            child: ConstrainedBox(
              constraints: BoxConstraints(minHeight: constraints.maxHeight),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  _buildHeader('Upload Source Image', 'Select the face you want to use'),
                  const SizedBox(height: 40),
                  _buildImagePreview(_sourceImage, 'No Image Selected'),
                  const SizedBox(height: 30),
                  _buildUploadButton(
                    'Select Source Image',
                    Icons.photo_library,
                    _pickSourceImage,
                  ),
                  if (_sourceImage != null) ...[
                    const SizedBox(height: 20),
                    _buildNextButton(() => setState(() => _currentStep = 1)),
                  ],
                  const SizedBox(height: 40),
                  _buildStepIndicator(0),
                ],
              ),
            ),
          );
        }
    );
  }

  Widget _buildUploadTargetStep() {
    return LayoutBuilder(
        builder: (context, constraints) {
          return SingleChildScrollView(
            padding: const EdgeInsets.all(24.0),
            child: ConstrainedBox(
              constraints: BoxConstraints(minHeight: constraints.maxHeight),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  _buildHeader('Upload Target Video', 'Select the video to apply face swap'),
                  const SizedBox(height: 40),
                  _buildVideoPreview(_targetVideo),
                  const SizedBox(height: 30),
                  _buildUploadButton(
                    'Select Target Video',
                    Icons.video_library,
                    _pickTargetVideo,
                  ),
                  if (_targetVideo != null) ...[
                    const SizedBox(height: 20),
                    _buildNextButton(() => setState(() => _currentStep = 2)),
                  ],
                  const SizedBox(height: 40),
                  _buildStepIndicator(1),
                ],
              ),
            ),
          );
        }
    );
  }

  Widget _buildModeSelectionStep() {
    return LayoutBuilder(
        builder: (context, constraints) {
          return SingleChildScrollView(
            padding: const EdgeInsets.all(24.0),
            child: ConstrainedBox(
              constraints: BoxConstraints(minHeight: constraints.maxHeight),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  _buildHeader('Choose Quality Mode', 'Select processing quality level'),
                  const SizedBox(height: 40),
                  _buildModeCard(
                    'Normal Mode',
                    '⚡ Fast Processing (3-5 minutes)\n📱 Good for social media\n🔄 Quick results',
                    'normal',
                  ),
                  const SizedBox(height: 20),
                  _buildModeCard(
                    'Best Mode',
                    '🎨 Professional Quality (15-20 minutes)\n✨ AI Enhancement with CodeFormer\n🏆 Publication ready',
                    'best',
                  ),
                  const SizedBox(height: 30),
                  _buildProcessButton(),
                  const SizedBox(height: 40),
                  _buildStepIndicator(2),
                ],
              ),
            ),
          );
        }
    );
  }

  Widget _buildProcessingStep() {
    Duration elapsed = _processStartTime != null
        ? DateTime.now().difference(_processStartTime!)
        : const Duration();
    String elapsedText = '${elapsed.inMinutes}:${(elapsed.inSeconds % 60).toString().padLeft(2, '0')}';

    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        children: [
          Text(
            'FAKESYNC STUDIO',
            style: TextStyle(
              fontSize: 28,
              fontWeight: FontWeight.bold,
              foreground: Paint()
                ..shader = LinearGradient(
                  colors: [Colors.purple, Colors.purple.shade300],
                ).createShader(const Rect.fromLTWH(0.0, 0.0, 200.0, 70.0)),
            ),
          ),
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            decoration: BoxDecoration(
              color: _selectedMode == 'best'
                  ? Colors.purple.withOpacity(0.2)
                  : Colors.blue.withOpacity(0.2),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                color: _selectedMode == 'best'
                    ? Colors.purple.withOpacity(0.5)
                    : Colors.blue.withOpacity(0.5),
              ),
            ),
            child: Text(
              '${_selectedMode.toUpperCase()} MODE',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: _selectedMode == 'best'
                    ? Colors.purple.shade300
                    : Colors.blue.shade300,
              ),
            ),
          ),
          const SizedBox(height: 40),

          Expanded(
            child: LayoutBuilder(
                builder: (context, constraints) {
                  return SingleChildScrollView(
                    child: ConstrainedBox(
                      constraints: BoxConstraints(minHeight: constraints.maxHeight),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const SizedBox(height: 20),
                          if (_isProcessing) ...[
                            AnimatedBuilder(
                              animation: _pulseAnimation,
                              builder: (context, child) {
                                return Transform.scale(
                                  scale: _pulseAnimation.value,
                                  child: Container(
                                    width: 120,
                                    height: 120,
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                      gradient: RadialGradient(
                                        colors: [
                                          Colors.purple.withOpacity(0.3),
                                          Colors.purple.withOpacity(0.1),
                                          Colors.transparent,
                                        ],
                                      ),
                                    ),
                                    child: const Center(
                                      child: CircularProgressIndicator(
                                        valueColor: AlwaysStoppedAnimation<Color>(Colors.purple),
                                        strokeWidth: 4.0,
                                      ),
                                    ),
                                  ),
                                );
                              },
                            ),
                          ] else ...[
                            Container(
                              width: 120,
                              height: 120,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                gradient: LinearGradient(
                                  colors: [
                                    Colors.purple.withOpacity(0.3),
                                    Colors.purple.withOpacity(0.1),
                                  ],
                                ),
                              ),
                              child: const Center(
                                child: CircularProgressIndicator(
                                  valueColor: AlwaysStoppedAnimation<Color>(Colors.purple),
                                  strokeWidth: 4.0,
                                ),
                              ),
                            ),
                          ],

                          const SizedBox(height: 30),

                          Text(
                            _isUploading ? 'Uploading Files' : 'Processing Face Swap',
                            style: const TextStyle(
                              fontSize: 24,
                              fontWeight: FontWeight.bold,
                              color: Colors.white,
                            ),
                            textAlign: TextAlign.center,
                          ),
                          const SizedBox(height: 10),

                          Text(
                            'Elapsed: $elapsedText',
                            style: TextStyle(
                              fontSize: 16,
                              color: Colors.grey.shade400,
                            ),
                          ),
                          const SizedBox(height: 20),

                          if (_isUploading && _uploadProgress < 1.0) ...[
                            Container(
                              width: double.infinity,
                              margin: const EdgeInsets.symmetric(horizontal: 20),
                              child: Column(
                                children: [
                                  LinearProgressIndicator(
                                    value: _uploadProgress,
                                    backgroundColor: Colors.grey.shade800,
                                    valueColor: AlwaysStoppedAnimation<Color>(Colors.purple),
                                  ),
                                  const SizedBox(height: 10),
                                  Text(
                                    '${(_uploadProgress * 100).toInt()}% - $_uploadStatus',
                                    style: TextStyle(
                                      fontSize: 14,
                                      color: Colors.grey.shade300,
                                    ),
                                    textAlign: TextAlign.center,
                                  ),
                                ],
                              ),
                            ),
                            const SizedBox(height: 20),
                          ],

                          Container(
                            margin: const EdgeInsets.symmetric(horizontal: 20),
                            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                            decoration: BoxDecoration(
                              color: Colors.purple.withOpacity(0.1),
                              borderRadius: BorderRadius.circular(25),
                              border: Border.all(color: Colors.purple.withOpacity(0.3)),
                            ),
                            child: Text(
                              _processingMessage,
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                color: Colors.purple.shade300,
                                fontSize: 14,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ),
                          const SizedBox(height: 20),

                          if (_isProcessing || _isUploading) ...[
                            Container(
                              margin: const EdgeInsets.symmetric(horizontal: 40),
                              width: double.infinity,
                              height: 50,
                              decoration: BoxDecoration(
                                color: Colors.red.shade600,
                                borderRadius: BorderRadius.circular(25),
                                boxShadow: [
                                  BoxShadow(
                                    color: Colors.red.withOpacity(0.3),
                                    blurRadius: 10,
                                    offset: const Offset(0, 5),
                                  ),
                                ],
                              ),
                              child: ElevatedButton.icon(
                                onPressed: () => _showCancelDialog(),
                                icon: const Icon(Icons.cancel, color: Colors.white, size: 20),
                                label: Text(
                                  _isUploading ? 'Cancel Upload' : 'Cancel Processing',
                                  style: const TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.w600,
                                    color: Colors.white,
                                  ),
                                ),
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: Colors.transparent,
                                  shadowColor: Colors.transparent,
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(25),
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(height: 20),
                          ],

                          Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 20),
                            child: Text(
                              _selectedMode == 'best'
                                  ? 'Applying AI enhancement and professional quality processing...'
                                  : 'Processing with fast mode for quick results...',
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                color: Colors.grey.shade500,
                                fontSize: 12,
                              ),
                            ),
                          ),
                          const SizedBox(height: 20),
                        ],
                      ),
                    ),
                  );
                }
            ),
          ),
        ],
      ),
    );
  }

  void _showCancelDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF2D2D2D),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        title: const Text(
          'Cancel Face Swap?',
          style: TextStyle(
            color: Colors.white,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        content: const Text(
          'Are you sure you want to cancel the face swap process? This action cannot be undone.',
          style: TextStyle(
            color: Colors.grey,
            fontSize: 16,
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text(
              'Continue Processing',
              style: TextStyle(color: Colors.grey),
            ),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              _cancelProcessing();
            },
            child: const Text(
              'Cancel Job',
              style: TextStyle(
                color: Colors.red,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeader(String title, String subtitle) {
    return Column(
      children: [
        Text(
          title,
          style: const TextStyle(
            fontSize: 28,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 8),
        Text(
          subtitle,
          style: TextStyle(
            fontSize: 16,
            color: Colors.grey.shade400,
          ),
          textAlign: TextAlign.center,
        ),
      ],
    );
  }

  Widget _buildImagePreview(File? image, String placeholderText) {
    return ConstrainedBox(
      constraints: const BoxConstraints(maxHeight: 250),
      child: Container(
        width: double.infinity,
        decoration: BoxDecoration(
          color: const Color(0xFF2D2D2D),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.purple.withOpacity(0.3)),
        ),
        child: image != null
            ? ClipRRect(
          borderRadius: BorderRadius.circular(16),
          child: Image.file(
            image,
            fit: BoxFit.cover,
          ),
        )
            : Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.image,
              size: 80,
              color: Colors.purple.withOpacity(0.6),
            ),
            const SizedBox(height: 16),
            Text(
              placeholderText,
              style: TextStyle(
                color: Colors.grey.shade400,
                fontSize: 16,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildVideoPreview(File? video) {
    return ConstrainedBox(
      constraints: const BoxConstraints(maxHeight: 250),
      child: Container(
        width: double.infinity,
        decoration: BoxDecoration(
          color: const Color(0xFF2D2D2D),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.purple.withOpacity(0.3)),
        ),
        child: video != null
            ? Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.video_file,
              size: 80,
              color: Colors.purple,
            ),
            const SizedBox(height: 16),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Text(
                video.path.split('/').last,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                ),
                textAlign: TextAlign.center,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        )
            : Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.videocam,
              size: 80,
              color: Colors.purple.withOpacity(0.6),
            ),
            const SizedBox(height: 16),
            Text(
              'No Video Selected',
              style: TextStyle(
                color: Colors.grey.shade400,
                fontSize: 16,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildUploadButton(String text, IconData icon, VoidCallback onPressed) {
    return SizedBox(
      width: double.infinity,
      height: 60,
      child: DecoratedBox(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.purple, Colors.purple.shade700],
          ),
          borderRadius: BorderRadius.circular(30.0),
          boxShadow: [
            BoxShadow(
              color: Colors.purple.withOpacity(0.3),
              blurRadius: 15,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: ElevatedButton.icon(
          onPressed: onPressed,
          icon: Icon(icon, color: Colors.white),
          label: Text(
            text,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w600,
              color: Colors.white,
            ),
          ),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.transparent,
            shadowColor: Colors.transparent,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(30.0),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildNextButton(VoidCallback onPressed) {
    return SizedBox(
      width: double.infinity,
      height: 50,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.purple.withOpacity(0.8),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(25.0),
          ),
        ),
        child: const Text(
          'Next',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: Colors.white,
          ),
        ),
      ),
    );
  }

  Widget _buildModeCard(String title, String description, String mode) {
    bool isSelected = _selectedMode == mode;
    return GestureDetector(
      onTap: () => setState(() => _selectedMode = mode),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        width: double.infinity,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: isSelected ? Colors.purple.withOpacity(0.2) : const Color(0xFF2D2D2D),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isSelected ? Colors.purple : Colors.purple.withOpacity(0.3),
            width: isSelected ? 2 : 1,
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  isSelected ? Icons.radio_button_checked : Icons.radio_button_unchecked,
                  color: Colors.purple,
                ),
                const SizedBox(width: 12),
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              description,
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey.shade300,
                height: 1.5,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildProcessButton() {
    return SizedBox(
      width: double.infinity,
      height: 60,
      child: DecoratedBox(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.purple, Colors.purple.shade700],
          ),
          borderRadius: BorderRadius.circular(30.0),
          boxShadow: [
            BoxShadow(
              color: Colors.purple.withOpacity(0.4),
              blurRadius: 20,
              offset: const Offset(0, 10),
            ),
          ],
        ),
        child: ElevatedButton(
          onPressed: _isUploading || _isProcessing ? null : _uploadFilesAndProcess,
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.transparent,
            shadowColor: Colors.transparent,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(30.0),
            ),
          ),
          child: _isUploading || _isProcessing
              ? const Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              SizedBox(
                width: 24,
                height: 24,
                child: CircularProgressIndicator(
                  color: Colors.white,
                  strokeWidth: 3.0,
                ),
              ),
              SizedBox(width: 16),
              Text(
                'Processing...',
                style: TextStyle(fontSize: 18, color: Colors.white),
              ),
            ],
          )
              : Text(
            'Start Face Swap (${_selectedMode == 'best' ? 'Best' : 'Normal'} Mode)',
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
            textAlign: TextAlign.center,
          ),
        ),
      ),
    );
  }

  Widget _buildStepIndicator(int currentStep) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: List.generate(3, (index) {
          return Container(
            margin: const EdgeInsets.symmetric(horizontal: 4),
            width: 12,
            height: 12,
            decoration: BoxDecoration(
              color: index <= currentStep ? Colors.purple : Colors.grey.shade600,
              shape: BoxShape.circle,
            ),
          );
        }),
      ),
    );
  }

  Widget _buildErrorMessage() {
    if (_errorMessage.isEmpty) return const SizedBox.shrink();

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.red.withOpacity(0.1),
        border: Border.all(color: Colors.red.withOpacity(0.3)),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        _errorMessage,
        style: const TextStyle(
          color: Colors.red,
          fontWeight: FontWeight.w500,
        ),
        textAlign: TextAlign.center,
      ),
    );
  }
}