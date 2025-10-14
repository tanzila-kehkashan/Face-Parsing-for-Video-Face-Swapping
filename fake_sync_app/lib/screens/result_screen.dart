import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:ui' as ui;
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:video_player/video_player.dart';
import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_new/return_code.dart';
import '../services/api_service.dart';
import '../services/navigation_service.dart';
import 'package:gal/gal.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

class ResultScreen extends StatefulWidget {
  final String uniqueId;

  const ResultScreen({super.key, required this.uniqueId});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> with TickerProviderStateMixin {
  final ApiService _apiService = ApiService();
  final Random _random = Random();
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  String _status = 'loading';
  bool _isDownloading = false;
  String _errorMessage = '';
  String _downloadMessage = 'Preparing download...';
  String? _userId;

  VideoPlayerController? _videoController;
  bool _isVideoInitialized = false;
  File? _videoFile;

  // Animation controllers
  late AnimationController _fadeController;
  late Animation<double> _fadeAnimation;

  // Floating icon animation
  late AnimationController _iconController;
  Animation<Offset>? _iconAnimation;
  bool _showFloatingIcon = false;

  // Displayed size of the video on screen
  Size? _displayedSize;
  final _videoContainerKey = GlobalKey();

  @override
  void initState() {
    super.initState();
    _setupAnimations();
    _fetchUserId();
    _downloadAndPlayVideo();
    _getDisplayedSize();
  }

  void _setupAnimations() {
    _fadeController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );
    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _fadeController, curve: Curves.easeInOut),
    );
    _fadeController.forward();

    _iconController = AnimationController(
      duration: const Duration(seconds: 15),
      vsync: this,
    )..addStatusListener((status) {
      if (status == AnimationStatus.completed) {
        _resetIconAnimation();
      }
    });
  }

  Future<void> _fetchUserId() async {
    try {
      final user = _auth.currentUser;
      if (user == null) {
        NavigationService.showErrorSnackBar('User not authenticated');
        return;
      }

      final doc = await _firestore.collection('user').doc(user.uid).get();
      if (doc.exists && mounted) {
        setState(() {
          _userId = doc.data()?['userId'] as String? ?? 'N/A';
        });
      } else {
        NavigationService.showErrorSnackBar('User data not found in Firestore');
      }
    } catch (e) {
      NavigationService.showErrorSnackBar('Failed to fetch user ID: $e');
    }
  }

  void _startIconAnimation() {
    if (_displayedSize == null) return;

    final maxX = _displayedSize!.width - 40;
    final maxY = _displayedSize!.height - 40;

    final start = Offset(
      _random.nextDouble() * maxX,
      _random.nextDouble() * maxY,
    );

    final end = Offset(
      _random.nextDouble() * maxX,
      _random.nextDouble() * maxY,
    );

    _iconAnimation = Tween<Offset>(
      begin: start,
      end: end,
    ).animate(CurvedAnimation(
      parent: _iconController,
      curve: Curves.easeInOut,
    ));

    setState(() {
      _showFloatingIcon = true;
    });

    _iconController
      ..reset()
      ..forward();
  }

  void _resetIconAnimation() {
    if (_iconController.status == AnimationStatus.completed) {
      _iconController.reset();
      if (mounted) {
        _startIconAnimation();
      }
    }
  }

  void _getDisplayedSize() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final renderBox = _videoContainerKey.currentContext?.findRenderObject() as RenderBox?;
      if (renderBox != null && _displayedSize == null) {
        setState(() {
          _displayedSize = renderBox.size;
        });
      } else if (_displayedSize == null) {
        _getDisplayedSize();
      }
    });
  }

  @override
  void dispose() {
    _videoController?.dispose();
    _fadeController.dispose();
    _iconController.dispose();
    super.dispose();
  }

  Future<void> _downloadAndPlayVideo() async {
    if (!mounted) return;

    setState(() {
      _isDownloading = true;
      _downloadMessage = 'Downloading your enhanced video...';
    });

    try {
      final File videoFile = await _apiService.downloadResult(widget.uniqueId);

      if (!await videoFile.exists()) {
        throw Exception('Downloaded video file does not exist');
      }

      final fileSize = await videoFile.length();
      if (fileSize == 0) {
        throw Exception('Downloaded video file is empty');
      }

      _videoFile = videoFile;
      _videoController = VideoPlayerController.file(videoFile);
      await _videoController!.initialize();

      if (!mounted) return;

      setState(() {
        _isVideoInitialized = true;
        _status = 'completed';
        _downloadMessage = 'Video ready! 🎉';
      });

      _videoController!
        ..play()
        ..addListener(() {
          if (_videoController!.value.isPlaying && !_showFloatingIcon && _displayedSize != null) {
            _startIconAnimation();
          }
        });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Failed to download or play video: $e';
        _status = 'failed';
      });
    } finally {
      if (!mounted) return;
      setState(() => _isDownloading = false);
    }
  }

  Future<String> getIconPath() async {
    final byteData = await rootBundle.load('assets/floating_icon.png');
    final tempDir = await getTemporaryDirectory();
    final file = File('${tempDir.path}/floating_icon.png');
    await file.writeAsBytes(byteData.buffer.asUint8List(byteData.offsetInBytes, byteData.lengthInBytes));
    return file.path;
  }

  Future<String> compositeVideo() async {
    if (_videoFile == null || !_isVideoInitialized) {
      throw Exception('Video file or controller not initialized');
    }

    if (_userId == null || _userId == 'N/A') {
      throw Exception('User ID not available for embedding in metadata');
    }

    String inputVideo = _videoFile!.path;
    String iconPath = await getIconPath();

    // Get video properties
    double durationSeconds = _videoController!.value.duration.inMilliseconds / 1000.0;
    int videoWidth = _videoController!.value.size.width.toInt();
    int videoHeight = _videoController!.value.size.height.toInt();

    // Get icon size
    final byteData = await rootBundle.load('assets/floating_icon.png');
    final codec = await ui.instantiateImageCodec(byteData.buffer.asUint8List());
    final frameInfo = await codec.getNextFrame();
    int iconWidth = frameInfo.image.width;
    int iconHeight = frameInfo.image.height;

    // Generate keyframe times
    List<double> times = [0];
    double t = 15.0;
    while (t <= durationSeconds) {
      times.add(t);
      t += 15.0;
    }
    if (times.last < durationSeconds) {
      times.add(durationSeconds);
    }

    // Generate random positions for each keyframe
    List<int> xs = [];
    List<int> ys = [];
    for (var time in times) {
      int maxX = videoWidth - iconWidth;
      int maxY = videoHeight - iconHeight;
      if (maxX < 0) maxX = 0;
      if (maxY < 0) maxY = 0;
      int x = (_random.nextDouble() * maxX).toInt();
      int y = (_random.nextDouble() * maxY).toInt();
      xs.add(x);
      ys.add(y);
    }

    // Build interpolate expressions
    String buildInterpolateExpr(List<double> times, List<int> values) {
      if (times.length < 2) return values[0].toString();
      int n = times.length;
      String expr = '${values[n-2]} + (${values[n-1]} - ${values[n-2]}) * ((t - ${times[n-2].toStringAsFixed(3)}) / (${times[n-1].toStringAsFixed(3)} - ${times[n-2].toStringAsFixed(3)}))';
      for (int i = n-3; i >= 0; i--) {
        String linear = '${values[i]} + (${values[i+1]} - ${values[i]}) * ((t - ${times[i].toStringAsFixed(3)}) / (${times[i+1].toStringAsFixed(3)} - ${times[i].toStringAsFixed(3)}))';
        expr = 'if(lt(t, ${times[i+1].toStringAsFixed(3)}), $linear, $expr)';
      }
      return expr;
    }

    String xExpr = buildInterpolateExpr(times, xs);
    String yExpr = buildInterpolateExpr(times, ys);

    // Define output path
    final tempDir = await getTemporaryDirectory();
    String outputVideo = '${tempDir.path}/composited_${DateTime.now().millisecondsSinceEpoch}.mp4';

    // Construct FFmpeg command with Firestore userId as job_id
    String command = '-i "$inputVideo" -i "$iconPath" -filter_complex "[0:v][1:v]overlay=x=$xExpr:y=$yExpr[out]" -map "[out]" -map 0:a -c:a copy -map_metadata 0 -metadata job_id="$_userId" "$outputVideo"';

    // Execute FFmpeg
    final session = await FFmpegKit.execute(command);
    final returnCode = await session.getReturnCode();
    if (ReturnCode.isSuccess(returnCode)) {
      return outputVideo;
    } else {
      throw Exception('FFmpeg compositing failed with code $returnCode');
    }
  }

  Future<Map<String, String>> extractMetadata(String videoPath) async {
    final tempDir = await getTemporaryDirectory();
    final tempFile = File('${tempDir.path}/metadata.txt');
    final command = '-i "$videoPath" -f ffmetadata "${tempFile.path}"';
    final session = await FFmpegKit.execute(command);
    final returnCode = await session.getReturnCode();
    if (ReturnCode.isSuccess(returnCode)) {
      final metadata = await tempFile.readAsString();
      await tempFile.delete();
      final lines = metadata.split('\n');
      final Map<String, String> metadataMap = {};
      for (var line in lines) {
        if (line.contains('=')) {
          final parts = line.split('=');
          if (parts.length == 2) {
            metadataMap[parts[0].trim()] = parts[1].trim();
          }
        }
      }
      return metadataMap;
    } else {
      throw Exception('Failed to extract metadata');
    }
  }

  void _showMetadata() async {
    if (_videoFile == null || !await _videoFile!.exists()) {
      NavigationService.showErrorSnackBar('Video file not available');
      return;
    }
    try {
      final metadata = await extractMetadata(_videoFile!.path);
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          backgroundColor: const Color(0xFF2D2D2D),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          title: const Text(
            'Video Metadata',
            style: TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          content: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'User ID:',
                  style: TextStyle(
                    color: Colors.purple,
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  _userId ?? 'Loading...',
                  style: const TextStyle(color: Colors.white),
                ),
                const SizedBox(height: 16),
                const Text(
                  'Video Metadata:',
                  style: TextStyle(
                    color: Colors.purple,
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                if (metadata.isEmpty)
                  const Text(
                    'No metadata found',
                    style: TextStyle(color: Colors.grey),
                  )
                else
                  ...metadata.entries.map((e) => Padding(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    child: Text(
                      '${e.key}: ${e.value}',
                      style: const TextStyle(color: Colors.white),
                    ),
                  )),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text(
                'Close',
                style: TextStyle(color: Colors.purple),
              ),
            ),
          ],
        ),
      );
    } catch (e) {
      NavigationService.showErrorSnackBar('Failed to extract metadata: $e');
    }
  }

  Future<void> _saveVideoToGallery() async {
    if (_videoFile == null || !await _videoFile!.exists() || !_isVideoInitialized) {
      NavigationService.showErrorSnackBar('Video file not available for saving');
      return;
    }

    try {
      setState(() {
        _downloadMessage = 'Compositing and saving video...';
      });

      String compositedVideo = await compositeVideo();
      bool hasAccess = await Gal.hasAccess();
      if (!hasAccess) {
        bool granted = await Gal.requestAccess();
        if (!granted) {
          NavigationService.showErrorSnackBar('Gallery access permission required');
          return;
        }
      }

      await Gal.putVideo(compositedVideo);
      NavigationService.showSuccessSnackBar('Video saved to gallery successfully!');
      await File(compositedVideo).delete();
    } catch (e) {
      print('Error compositing or saving video: $e');
      String errorMessage = 'Failed to save video';
      if (e.toString().contains('format')) {
        errorMessage = 'Video format not supported by gallery. Try sharing instead.';
      } else if (e.toString().contains('permission')) {
        errorMessage = 'Permission denied. Please check gallery permissions.';
      } else if (e.toString().contains('space')) {
        errorMessage = 'Not enough storage space to save video.';
      } else {
        errorMessage = 'Failed to save video: ${e.toString()}';
      }
      NavigationService.showErrorSnackBar(errorMessage);
      _showSaveAlternativeDialog();
    } finally {
      if (mounted) {
        setState(() {
          _downloadMessage = 'Video ready! 🎉';
        });
      }
    }
  }

  Future<void> _shareVideo() async {
    if (_videoFile == null || !await _videoFile!.exists() || !_isVideoInitialized) {
      NavigationService.showErrorSnackBar('Video file not available for sharing');
      return;
    }

    try {
      setState(() {
        _downloadMessage = 'Compositing video...';
      });

      String compositedVideo = await compositeVideo();
      await Share.shareXFiles(
        [XFile(compositedVideo)],
        text: 'Check out my face swap video created with FakeSync Studio!',
        subject: 'Face Swap Video',
      );
      await File(compositedVideo).delete();
    } catch (e) {
      print('Error compositing or sharing video: $e');
      NavigationService.showErrorSnackBar('Failed to share video: $e');
    } finally {
      if (mounted) {
        setState(() {
          _downloadMessage = 'Video ready! 🎉';
        });
      }
    }
  }

  void _showSaveAlternativeDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF2D2D2D),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        title: const Text(
          'Save Alternative',
          style: TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
        content: const Text(
          'Unable to save directly to gallery. Would you like to share the video instead? You can then save it from your share options.',
          style: TextStyle(
            color: Colors.grey,
            fontSize: 16,
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text(
              'Cancel',
              style: TextStyle(color: Colors.grey),
            ),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              _shareVideo();
            },
            child: const Text(
              'Share Video',
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

  Widget _buildActionButton(
      String label,
      IconData icon,
      Color color,
      VoidCallback onPressed,
      double width,
      ) {
    return SizedBox(
      width: width,
      child: ElevatedButton(
        style: ElevatedButton.styleFrom(
          backgroundColor: color,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          padding: const EdgeInsets.symmetric(vertical: 16),
        ),
        onPressed: onPressed,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: Colors.white, size: 20),
            const SizedBox(width: 8),
            Text(
              label,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
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
            child: _buildContent(),
          ),
        ),
      ),
    );
  }

  Widget _buildContent() {
    if (_isDownloading) {
      return _buildDownloadingView();
    } else if (_status == 'completed' && _isVideoInitialized) {
      return _buildCompletedView();
    } else if (_status == 'failed') {
      return _buildFailedView();
    } else {
      return _buildDownloadingView();
    }
  }

  Widget _buildDownloadingView() {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        children: [
          _buildHeader(),
          const SizedBox(height: 40),
          Expanded(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
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
                const SizedBox(height: 30),
                const Text(
                  'Preparing Result',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 20),
                Container(
                  margin: const EdgeInsets.symmetric(horizontal: 20),
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                  decoration: BoxDecoration(
                    color: Colors.purple.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(25),
                    border: Border.all(color: Colors.purple.withOpacity(0.3)),
                  ),
                  child: Text(
                    _downloadMessage,
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: Colors.purple.shade300,
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  child: Text(
                    'Your face swap is complete! We are preparing the video for viewing.',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: Colors.grey.shade500,
                      fontSize: 12,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCompletedView() {
    return LayoutBuilder(
      builder: (context, constraints) {
        final isSmallScreen = constraints.maxWidth < 600;
        return Padding(
          padding: EdgeInsets.all(MediaQuery.of(context).size.width * 0.05),
          child: Column(
            children: [
              _buildHeader(showInfoButton: true),
              SizedBox(height: MediaQuery.of(context).size.height * 0.03),
              Container(
                width: 80,
                height: 80,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: LinearGradient(
                    colors: [Colors.green.shade400, Colors.green.shade600],
                  ),
                ),
                child: const Icon(
                  Icons.check,
                  color: Colors.white,
                  size: 40,
                ),
              ),
              SizedBox(height: MediaQuery.of(context).size.height * 0.03),
              const Text(
                'Face Swap Complete! 🎉',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
              SizedBox(height: MediaQuery.of(context).size.height * 0.04),
              Expanded(
                child: Container(
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(16),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.3),
                        blurRadius: 20,
                        offset: const Offset(0, 10),
                      ),
                    ],
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(16),
                    child: AspectRatio(
                      aspectRatio: _videoController!.value.aspectRatio,
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          Container(
                            key: _videoContainerKey,
                            child: VideoPlayer(_videoController!),
                          ),
                          if (_showFloatingIcon && _iconAnimation != null)
                            AnimatedBuilder(
                              animation: _iconAnimation!,
                              builder: (context, child) {
                                return Positioned(
                                  left: _iconAnimation!.value.dx,
                                  top: _iconAnimation!.value.dy,
                                  child: Image.asset(
                                    'assets/floating_icon.png',
                                    width: 40,
                                    height: 40,
                                    color: Colors.white.withOpacity(0.8),
                                  ),
                                );
                              },
                            ),
                          Center(
                            child: GestureDetector(
                              onTap: () {
                                setState(() {
                                  if (_videoController!.value.isPlaying) {
                                    _videoController!.pause();
                                  } else {
                                    _videoController!.play();
                                    if (!_showFloatingIcon && _displayedSize != null) {
                                      _startIconAnimation();
                                    }
                                  }
                                });
                              },
                              child: AnimatedOpacity(
                                opacity: _videoController!.value.isPlaying ? 0.0 : 1.0,
                                duration: const Duration(milliseconds: 300),
                                child: Container(
                                  width: 80,
                                  height: 80,
                                  decoration: BoxDecoration(
                                    color: Colors.black.withOpacity(0.7),
                                    shape: BoxShape.circle,
                                  ),
                                  child: const Icon(
                                    Icons.play_arrow,
                                    size: 40,
                                    color: Colors.white,
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
              SizedBox(height: MediaQuery.of(context).size.height * 0.04),
              isSmallScreen
                  ? Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  _buildActionButton(
                    'Save to Gallery',
                    Icons.download,
                    Colors.purple,
                    _saveVideoToGallery,
                    constraints.maxWidth,
                  ),
                  SizedBox(height: MediaQuery.of(context).size.height * 0.02),
                  _buildActionButton(
                    'Share Video',
                    Icons.share,
                    Colors.blue.shade600,
                    _shareVideo,
                    constraints.maxWidth,
                  ),
                  SizedBox(height: MediaQuery.of(context).size.height * 0.02),
                  _buildActionButton(
                    'New Swap',
                    Icons.refresh,
                    Colors.grey.shade700,
                        () => NavigationService.goToHome(),
                    constraints.maxWidth,
                  ),

                ],
              )
                  : Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Expanded(
                    child: _buildActionButton(
                      'Save to Gallery',
                      Icons.download,
                      Colors.purple,
                      _saveVideoToGallery,
                      constraints.maxWidth / 2 - 16,
                    ),
                  ),
                  SizedBox(width: MediaQuery.of(context).size.width * 0.04),
                  Expanded(
                    child: _buildActionButton(
                      'Share Video',
                      Icons.share,
                      Colors.blue.shade600,
                      _shareVideo,
                      constraints.maxWidth / 2 - 16,
                    ),
                  ),
                ],
              ),
              if (!isSmallScreen) SizedBox(height: MediaQuery.of(context).size.height * 0.02),
              if (!isSmallScreen)
                _buildActionButton(
                  'New Swap',
                  Icons.refresh,
                  Colors.grey.shade700,
                      () => NavigationService.goToHome(),
                  constraints.maxWidth,
                ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildFailedView() {
    return Padding(
      padding: EdgeInsets.all(MediaQuery.of(context).size.width * 0.05),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 80,
            height: 80,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: LinearGradient(
                colors: [Colors.red.shade400, Colors.red.shade600],
              ),
            ),
            child: const Icon(
              Icons.error_outline,
              color: Colors.white,
              size: 40,
            ),
          ),
          SizedBox(height: MediaQuery.of(context).size.height * 0.03),
          const Text(
            'Download Failed',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
          SizedBox(height: MediaQuery.of(context).size.height * 0.02),
          if (_errorMessage.isNotEmpty)
            Container(
              padding: const EdgeInsets.all(16),
              margin: EdgeInsets.symmetric(vertical: MediaQuery.of(context).size.height * 0.03),
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
            ),
          SizedBox(height: MediaQuery.of(context).size.height * 0.04),
          _buildActionButton(
            'Try Again',
            Icons.refresh,
            Colors.purple,
            _downloadAndPlayVideo,
            MediaQuery.of(context).size.width,
          ),
          SizedBox(height: MediaQuery.of(context).size.height * 0.03),
          _buildActionButton(
            'Back to Home',
            Icons.home,
            Colors.grey.shade700,
                () => NavigationService.goToHome(),
            MediaQuery.of(context).size.width,
          ),
        ],
      ),
    );
  }

  Widget _buildHeader({bool showInfoButton = false}) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
        const Text(
          'Result',
          style: TextStyle(
            color: Colors.white,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        showInfoButton
            ? IconButton(
          icon: const Icon(Icons.info_outline, color: Colors.purple),
          onPressed: _showMetadata,
        )
            : const SizedBox(width: 48), // Placeholder for alignment
      ],
    );
  }
}