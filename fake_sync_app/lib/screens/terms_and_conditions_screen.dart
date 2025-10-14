import 'package:flutter/material.dart';
import '../services/navigation_service.dart';

class TermsAndConditionsScreen extends StatefulWidget {
  const TermsAndConditionsScreen({super.key});

  @override
  State<TermsAndConditionsScreen> createState() => _TermsAndConditionsScreenState();
}

class _TermsAndConditionsScreenState extends State<TermsAndConditionsScreen> with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<Offset> _slideAnimation;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );

    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, 0.05),
      end: Offset.zero,
    ).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeOutQuart),
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
            child: SlideTransition(
              position: _slideAnimation,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    // Close button at top right
                    Align(
                      alignment: Alignment.topRight,
                      child: IconButton(
                        icon: Icon(
                          Icons.close,
                          color: Colors.grey.shade400,
                          size: 28,
                        ),
                        onPressed: () => Navigator.pop(context),
                      ),
                    ),

                    // Header with icon
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 8.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.description,
                            color: Colors.purple.shade300,
                            size: 28,
                          ),
                          const SizedBox(width: 12),
                          const Text(
                            'TERMS & CONDITIONS',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.w600,
                              color: Colors.white,
                              letterSpacing: 0.5,
                            ),
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 10),

                    // Terms content container
                    Expanded(
                      child: Container(
                        decoration: BoxDecoration(
                          color: const Color(0xFF2A2A2A),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(
                            color: Colors.purple.withOpacity(0.3),
                            width: 1,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.4),
                              blurRadius: 12,
                              offset: const Offset(0, 6),
                            ),
                          ],
                        ),
                        child: SingleChildScrollView(
                          padding: const EdgeInsets.all(20.0),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              _buildSectionHeader('1. Acceptance of Terms'),
                              _buildSectionText('By accessing or using FAKESYNC STUDIO, you agree to be bound by these Terms and Conditions.'),

                              _buildSectionHeader('2. User Responsibilities'),
                              _buildSectionText('You are responsible for maintaining the confidentiality of your account credentials and for all activities that occur under your account.'),

                              _buildSectionHeader('3. Prohibited Activities'),
                              _buildSectionText('You agree not to use the service for any illegal purposes, harassment, infringement of intellectual property rights, or distribution of harmful content.'),

                              _buildSectionHeader('4. Intellectual Property'),
                              _buildSectionText('All content, trademarks, and data on this platform, including software, text, and graphics, are owned by FAKESYNC STUDIO and protected by intellectual property laws.'),

                              _buildSectionHeader('5. Limitation of Liability'),
                              _buildSectionText('FAKESYNC STUDIO shall not be liable for any indirect, incidental, or consequential damages arising from your use of the service.'),

                              _buildSectionHeader('6. Changes to Terms'),
                              _buildSectionText('We reserve the right to modify these terms at any time. Continued use after changes constitutes acceptance of the new terms.'),

                              _buildSectionHeader('7. Governing Law'),
                              _buildSectionText('These terms shall be governed by and construed in accordance with the laws of the jurisdiction where FAKESYNC STUDIO is established.'),

                              const SizedBox(height: 20),
                              Container(
                                padding: const EdgeInsets.all(16),
                                decoration: BoxDecoration(
                                  color: Colors.purple.withOpacity(0.1),
                                  borderRadius: BorderRadius.circular(12),
                                  border: Border.all(
                                    color: Colors.purple.withOpacity(0.3),
                                  ),
                                ),
                                child: const Text(
                                  'Please return to the signup screen and check the box to accept these terms before proceeding.',
                                  style: TextStyle(
                                    fontSize: 14,
                                    color: Colors.purpleAccent,
                                    fontWeight: FontWeight.w500,
                                  ),
                                  textAlign: TextAlign.center,
                                ),
                              ),
                            ],
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
      ),
    );
  }

  Widget _buildSectionHeader(String text) {
    return Padding(
      padding: const EdgeInsets.only(top: 15, bottom: 8),
      child: Text(
        text,
        style: TextStyle(
          fontSize: 17,
          fontWeight: FontWeight.w600,
          color: Colors.purple.shade300,
        ),
      ),
    );
  }

  Widget _buildSectionText(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Text(
        text,
        style: const TextStyle(
          fontSize: 15,
          color: Colors.white70,
          height: 1.5,
        ),
      ),
    );
  }
}