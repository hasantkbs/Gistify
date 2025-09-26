import 'package:flutter/material.dart';

import '../../theme/app_colors.dart';

class NotificationsPage extends StatefulWidget {
  const NotificationsPage({super.key});

  static const route = '/notifications';

  @override
  State<NotificationsPage> createState() => _NotificationsPageState();
}

class _NotificationsPageState extends State<NotificationsPage> {
  bool emailAlerts = true;
  bool pushAlerts = true;
  bool summaryDigest = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(24, 16, 24, 32),
          children: [
            _header(context, 'Notifications'),
            const SizedBox(height: 24),
            _toggleTile(
              icon: Icons.email_outlined,
              title: 'Email alerts',
              subtitle: 'Receive summary updates via email',
              value: emailAlerts,
              onChanged: (value) => setState(() => emailAlerts = value),
            ),
            _toggleTile(
              icon: Icons.notifications_none,
              title: 'Push notifications',
              subtitle: 'Stay informed about changes instantly',
              value: pushAlerts,
              onChanged: (value) => setState(() => pushAlerts = value),
            ),
            _toggleTile(
              icon: Icons.calendar_today_outlined,
              title: 'Weekly digest',
              subtitle: 'Summary of highlights every Monday',
              value: summaryDigest,
              onChanged: (value) => setState(() => summaryDigest = value),
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
