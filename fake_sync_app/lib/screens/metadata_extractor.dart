import 'dart:io';
import 'package:flutter/material.dart';
import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_new/return_code.dart';
import 'package:file_picker/file_picker.dart';
import 'package:path_provider/path_provider.dart';
import '../services/navigation_service.dart';

class VideoMetadataScreen extends StatefulWidget {
  const VideoMetadataScreen({super.key});

  @override
  State<VideoMetadataScreen> createState() => _VideoMetadataScreenState();
}

class _VideoMetadataScreenState extends State<VideoMetadataScreen> with SingleTickerProviderStateMixin {
  File? _selectedVideo;
  String? _userId;
  Map<String, String> _metadata = {};
  bool _isLoading = false;
  String _statusMessage = 'Select a video to view metadata';
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );
    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeInOut),
    );
    _animationController.forward();
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  Future<void> _pickVideo() async {
    setState(() {
      _isLoading = true;
      _statusMessage = 'Selecting video...';
    });

    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.video,
        allowMultiple: false,
      );

      if (result != null && result.files.single.path != null) {
        _selectedVideo = File(result.files.single.path!);
        await _extractMetadata();
      } else {
        setState(() {
          _statusMessage = 'No video selected';
          _selectedVideo = null;
          _metadata = {};
          _userId = null;
        });
      }
    } catch (e) {
      NavigationService.showErrorSnackBar('Failed to select video: $e');
      setState(() {
        _statusMessage = 'Error selecting video';
      });
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _extractMetadata() async {
    if (_selectedVideo == null || !await _selectedVideo!.exists()) {
      NavigationService.showErrorSnackBar('Video file not available');
      setState(() {
        _statusMessage = 'Video file not available';
        _metadata = {};
        _userId = null;
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _statusMessage = 'Extracting metadata...';
    });

    try {
      final tempDir = await getTemporaryDirectory();
      final tempFile = File('${tempDir.path}/metadata.txt');
      final command = '-i "${_selectedVideo!.path}" -f ffmetadata "${tempFile.path}"';
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

        setState(() {
          _metadata = metadataMap;
          _userId = metadataMap['job_id'] ?? 'N/A';
          _statusMessage = 'Metadata extracted successfully!';
        });
      } else {
        throw Exception('FFmpeg metadata extraction failed with code $returnCode');
      }
    } catch (e) {
      NavigationService.showErrorSnackBar('Failed to extract metadata: $e');
      setState(() {
        _statusMessage = 'Failed to extract metadata';
        _metadata = {};
        _userId = null;
      });
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Widget _buildHeader() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
        const Text(
          'Video Metadata',
          style: TextStyle(
            color: Colors.white,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(width: 48), // Placeholder for alignment
      ],
    );
  }

  Widget _buildActionButton(
      String label,
      IconData icon,
      Color color,
      VoidCallback onPressed,
      ) {
    return Container(
      width: double.infinity,
      height: 52,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(12),
        gradient: LinearGradient(
          colors: [
            color.withOpacity(0.8),
            color,
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.3),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: ElevatedButton.icon(
        onPressed: onPressed,
        icon: Icon(icon, color: Colors.white, size: 20),
        label: Text(
          label,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.transparent,
          shadowColor: Colors.transparent,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
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
            child: Padding(
              padding: EdgeInsets.all(MediaQuery.of(context).size.width * 0.05),
              child: Column(
                children: [
                  _buildHeader(),
                  const SizedBox(height: 40),
                  Expanded(
                    child: _isLoading
                        ? Column(
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
                        const SizedBox(height: 20),
                        Text(
                          _statusMessage,
                          style: TextStyle(
                            color: Colors.purple.shade300,
                            fontSize: 14,
                            fontWeight: FontWeight.w500,
                          ),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    )
                        : SingleChildScrollView(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Container(
                            padding: const EdgeInsets.all(20),
                            decoration: BoxDecoration(
                              color: const Color(0xFF2A2A2A),
                              borderRadius: BorderRadius.circular(16),
                              border: Border.all(color: Colors.purple.withOpacity(0.3)),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black.withOpacity(0.3),
                                  blurRadius: 10,
                                  offset: const Offset(0, 5),
                                ),
                              ],
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Icon(Icons.video_file, color: Colors.purple.shade300, size: 24),
                                    const SizedBox(width: 12),
                                    Expanded(
                                      child: Text(
                                        _selectedVideo != null
                                            ? _selectedVideo!.path.split('/').last
                                            : 'No video selected',
                                        style: TextStyle(
                                          color: Colors.white,
                                          fontSize: 16,
                                          fontWeight: FontWeight.w600,
                                        ),
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 20),
                                _buildActionButton(
                                  'Select Video',
                                  Icons.upload,
                                  Colors.purple,
                                  _pickVideo,
                                ),
                              ],
                            ),
                          ),
                          const SizedBox(height: 30),
                          if (_selectedVideo != null) ...[
                            Container(
                              padding: const EdgeInsets.all(20),
                              decoration: BoxDecoration(
                                color: const Color(0xFF2A2A2A),
                                borderRadius: BorderRadius.circular(16),
                                border: Border.all(color: Colors.purple.withOpacity(0.3)),
                                boxShadow: [
                                  BoxShadow(
                                    color: Colors.black.withOpacity(0.3),
                                    blurRadius: 10,
                                    offset: const Offset(0, 5),
                                  ),
                                ],
                              ),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Row(
                                    children: [
                                      Icon(Icons.info_outline, color: Colors.purple.shade300, size: 24),
                                      const SizedBox(width: 12),
                                      Text(
                                        'METADATA',
                                        style: TextStyle(
                                          fontSize: 16,
                                          fontWeight: FontWeight.w600,
                                          color: Colors.purple.shade300,
                                          letterSpacing: 0.8,
                                        ),
                                      ),
                                    ],
                                  ),
                                  const SizedBox(height: 20),
                                  Text(
                                    'User ID:',
                                    style: TextStyle(
                                      color: Colors.purple.shade300,
                                      fontSize: 15,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  const SizedBox(height: 8),
                                  Text(
                                    _userId ?? 'N/A',
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 15,
                                      fontWeight: FontWeight.w500,
                                    ),
                                  ),
                                  const SizedBox(height: 20),
                                  Text(
                                    'Video Metadata:',
                                    style: TextStyle(
                                      color: Colors.purple.shade300,
                                      fontSize: 15,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  const SizedBox(height: 8),
                                  if (_metadata.isEmpty)
                                    Text(
                                      'No metadata found',
                                      style: TextStyle(
                                        color: Colors.grey.shade400,
                                        fontSize: 15,
                                      ),
                                    )
                                  else
                                    ..._metadata.entries.map((e) => Padding(
                                      padding: const EdgeInsets.symmetric(vertical: 8),
                                      child: Row(
                                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                        children: [
                                          Text(
                                            e.key,
                                            style: TextStyle(
                                              color: Colors.grey.shade400,
                                              fontSize: 15,
                                            ),
                                          ),
                                          const SizedBox(width: 16),
                                          Expanded(
                                            child: Text(
                                              e.value,
                                              style: const TextStyle(
                                                color: Colors.white,
                                                fontSize: 15,
                                                fontWeight: FontWeight.w500,
                                              ),
                                              textAlign: TextAlign.right,
                                            ),
                                          ),
                                        ],
                                      ),
                                    )),
                                ],
                              ),
                            ),
                          ],
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 20),
                  _buildActionButton(
                    'Back to Home',
                    Icons.home,
                    Colors.grey.shade700,
                        () => NavigationService.goToHome(),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}