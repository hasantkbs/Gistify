import 'package:flutter/material.dart';

import '../../theme/app_colors.dart';

class PrivacySecurityPage extends StatefulWidget {
  const PrivacySecurityPage({super.key});

  static const route = '/privacy-security';

  @override
  State<PrivacySecurityPage> createState() => _PrivacySecurityPageState();
}

class _PrivacySecurityPageState extends State<PrivacySecurityPage> {
  bool twoFactorEnabled = true;
  bool biometricEnabled = true;
  bool dataDownloadEnabled = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(24, 16, 24, 32),
          children: [
            _header(context, 'Privacy & Security'),
            const SizedBox(height: 24),
            _toggleTile(
              icon: Icons.verified_user_outlined,
              title: 'Two-factor authentication',
              subtitle: 'Add an extra layer of security to your account.',
              value: twoFactorEnabled,
              onChanged: (value) => setState(() => twoFactorEnabled = value),
            ),
            _toggleTile(
              icon: Icons.fingerprint,
              title: 'Biometric unlock',
              subtitle: 'Use Face ID or Touch ID to unlock the app.',
              value: biometricEnabled,
              onChanged: (value) => setState(() => biometricEnabled = value),
            ),
            _toggleTile(
              icon: Icons.download_outlined,
              title: 'Download data',
              subtitle: 'Receive a copy of your stored data.',
              value: dataDownloadEnabled,
              onChanged: (value) => setState(() => dataDownloadEnabled = value),
            ),
            const SizedBox(height: 24),
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: const [
                  Text(
                    'Privacy Tips',
                    style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16),
                  ),
                  SizedBox(height: 8),
                  Text(
                    'Review connected devices regularly and enable alerts to stay informed about suspicious activity.',
                    style: TextStyle(color: AppColors.textSecondary),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _toggleTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required bool value,
    required ValueChanged<bool> onChanged,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          children: [
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: AppColors.primaryBlue.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Icon(icon, color: AppColors.primaryBlue),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontWeight: FontWeight.w700,
                      fontSize: 16,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    subtitle,
                    style: const TextStyle(color: AppColors.textSecondary),
                  ),
                ],
              ),
            ),
            Switch.adaptive(value: value, onChanged: onChanged),
          ],
        ),
      ),
    );
  }

  Widget _header(BuildContext context, String title) {
    return Row(
      children: [
        IconButton(
          onPressed: () => Navigator.pop(context),
          icon: const Icon(Icons.arrow_back_ios_new),
        ),
        const Spacer(),
        Text(
          title,
          style: const TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w700,
            color: AppColors.textPrimary,
          ),
        ),
        const Spacer(),
        const SizedBox(width: 48),
      ],
    );
  }
}
